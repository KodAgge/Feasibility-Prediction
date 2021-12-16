import random
import typing
import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm
from time import time
from tabulate import tabulate

from scipy.stats import t
from scipy.special import comb

from utils.metrics import *
from utils.import_data import *
from utils.roc_helper import ROCObject, predict_proba

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


class ModelTester():
    """ModelTester is an object which accepts data and a ML model to analyze 
    given the data. 
    
    Parameters
    ----------
    x : pandas DataFrame
        samples
    y : pandas DataFrame
        labels
    model_function : callable
        function used to created model object, e.g. SVC
    model_parameters : keywords
        parameters for the given model, e.g. C = 1, gamma = 0.001
    """
    def __init__(self, x, y, model_function, **model_parameters):

        self.x = x
        self.y = y
        self.model = model_function
        self.parameters = model_parameters

        self.iter_size = 1024                                       # size of an iteration
        self.n_samples = int(self.x.shape[0])                   
        self.n_iterations = int(self.n_samples / self.iter_size)    # the number of iterations in the data

        self.iter_lag = 0                                           # how many iterations there is between the training and test data

        self.default_n_epohcs = 10                                  # the default number of times the model is trained

        # No custom functions at the start
        self.custom_train_model = None
        self.custom_prediction = None
        self.custom_predict_proba = None

        
    def training_loop(self, n_train = 8, n_test = 2, 
                      iter_lags = [0, 1, 2], sampling = 'under', 
                      starting_iters = False, seed = 0, 
                      printing = False) -> None:
        """Training loops 
        
        Parameters
        ----------
        n_train : int
            The number of iterations included in the training data. The default is 8.
        n_test : int
            The number of iterations included in the test data. The default is 2.
        iter_lags : list of int
            The lags to use. The default is [0, 1, 2].
        sampling : 'over','under'
            How the data should be balanced, 'under' = undersampling, 'over' = oversampling, else no balancing. The default is 'under'.
        starting_iters : boolean
            A list of from where the data should ba taken from. The default is False.
        seed : int
            Randomizer. The default is 0.
        transformer_class : function
            Dimensionality reduction class. E.g. PCA from sklearn or pytorch autoencode
            
        Returns
        -------
        None.
        """
        # If no starting_iters are given they are randomized
        if starting_iters == False:
            random.seed(seed)
            starting_iters = random.sample(range(self.n_iterations - n_train - n_test - int(sorted(iter_lags)[-1]) + 1), self.default_n_epohcs)
            self.starting_iters = starting_iters

        # Calculate counting variables
        n_runs = len(iter_lags)
        n_epochs = len(starting_iters)

        # Initialize metric objects
        metrics_test = MetricObject(n_runs, n_epochs)
        metrics_train = MetricObject(n_runs, n_epochs)

        # Initialize ROC objects
        roc_test = ROCObject(n_runs, n_epochs)
        roc_train = ROCObject(n_runs, n_epochs)

        # Save lags
        self.iter_lags = np.array(iter_lags)

        # The training loop
        for i, starting_iter in enumerate(starting_iters):

            if printing:
                print('Starting_iter ' + str(starting_iter) + ':')

            # Select and sample data
            X_train, y_train, X_test, y_test = self.select_data(starting_iter, n_train, iter_lags[0], n_test, sampling, seed)
            
            # Do standardization
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            # Train the model
            start_train = time()

            trained_model = self.train_model(
                X_train_transformed if transformer else X_train, 
                y_train
            )

            duration_train = time() - start_train

            for j, iter_lag in enumerate(iter_lags):

                # Select test data
                if j > 0:
                    _, _, X_test, y_test = self.select_data(starting_iter, n_train, iter_lag, n_test, sampling, seed, transformer)
                    X_test = sc.transform(X_test)
                # Make predictions
                start_predict_test = time()
                y_test_pred = self.prediction(
                    trained_model, 
                    X_test
                )
                duration_predict_test = time() - start_predict_test
                
                start_predict_train = time()
                y_train_pred = self.prediction(
                    trained_model, 
                    X_train
                )
                duration_predict_train = time() - start_predict_train

                # Update the metric objects
                metrics_test.update_metrics(y_test, y_test_pred, iter_lag, duration_train + duration_predict_test)
                metrics_train.update_metrics(y_train, y_train_pred, iter_lag, duration_train + duration_predict_train)

                # Update the ROC objects
                pptr = self.predict_probability(
                    X_train,
                    trained_model
                )
                ppte = self.predict_probability(
                    X_test,
                    trained_model
                )
                roc_test.update_rates(
                    y_test, 
                    ppte
                )
                roc_train.update_rates(
                    y_train,
                    pptr
                )

        # Save the metric objects
        self.metrics_test = metrics_test
        self.metrics_train = metrics_train

        # Save the ROC objects
        self.roc_test = roc_test
        self.roc_train = roc_train


    def set_custom_functions(self, custom_train_model = None, 
                             custom_prediction = None, 
                             custom_predict_proba = None) -> None:
        """Set custom functions.
        
        custom_train_model : callable
            Function for training the model. E.g. for pytorch models. The default is None.
        custom_prediction : callable
            Function for classifying samples. The default is None.
        custom_predict_proba : callable
            Function for predicting probabilities. The default is None.
        """
        self.custom_train_model = custom_train_model
        self.custom_prediction = custom_prediction
        self.custom_predict_proba = custom_predict_proba

    
    def predict_probability(self, x, trained_model) -> np.ndarray:
        """This function predicts probabilities *once*
        
        Parameters
        ----------
        x : np.ndarray
            Feature set. No default.
        trained_model : callable
            The trained model. No default.
        """
        # Predicting using the standard function
        if self.custom_predict_proba == None:
            return predict_proba(x, trained_model)

        # Predicting using the custom function
        else:
            return self.custom_predict_proba(self, x, trained_model)


    def train_model(self, x, y) -> typing.Callable:
        """This functions trains the model *once*

        Parameters
        ----------
        x : numpy array
            Features/independent variables. No default.
        y : numpy array 
            Explanatory/dependent variable. No default.

        Returns
        -------
        model : class
            Fitted model
        """

        # Training using the standard function
        if self.custom_train_model == None:
            model = self.model(**self.parameters)
            model.fit(x, y)

        # Training using the custom function
        else:
            model = self.custom_train_model(self, x, y)

        return model


    def select_data(self, starting_iter, n_train,
                    iter_lag, n_test, seed, 
                    sampling = 'under') -> typing.Sequence:
        """Returns test and train data given the chosen sampling.
        
        If it is 'under' or 'over' n_train is adjusted to still return the wanted number of samples.

        Parameters
        ----------
        starting_iter : int
            From where the data should be selected.
        n_train : int
            The number of iterations included in the training data.
        iter_lag : int
            Lag between the training iteration and test iteration.
        n_test : int
            The number of iterations included in the test data.
        seed : int
            Randomizer
        sampling : string, optional
            How the data should be balanced, 'under' = undersampling, 'over' = oversampling, else no balancing. The default is 'under'.

        Returns
        -------
        X_train : pandas DataFrame
            training sample.
        y_train : pandas DataFrame
            training labels.
        X_test : pandas DataFrame
            test sample.
        y_test : pandas DataFrame
            test labels.
        """
        
        # Adjusting n_train to still return the wanted number of samples
        if sampling == 'under':
            n_train *= 2
        elif sampling == 'over':
            n_train = int(n_train * 2 / 3 + 1)

        # Selecting the wanted iterations
        X_train, y_train, X_test, y_test = selectIterations(self.x, self.y, starting_iter, n_train, iter_lag, n_test, printing = False)

        # Under/over-sampling
        if sampling == 'under':
            X_train, y_train = underSampleNumpyArray(X_train, y_train, seed = seed, printing = False)
        elif sampling == 'over':
            X_train, y_train = overSampleNumpyArray(X_train, y_train, seed = seed)
            
        return X_train, y_train, X_test, y_test


    def prediction(self, model, x) -> np.ndarray:
        """ This function makes a prediction using the model

        Parameters
        --------
        model : class
            The trained model object
        x : numpy array
            Sample from which predictions are made
        
        Returns
        -------
        numpy.ndarray:
            predictions mad  by the model
        """
        # Predicting using the standard function
        if self.custom_prediction == None:
            return model.predict(x)

        # Predicting using the custom function
        else:
            return self.custom_prediction(self, x, model)


    def print_essentials_helper(self, values, variable_names, 
                                test = True, alpha = 0.95, n_sf = 2,
                                results_path = None) -> None:
        """A helper function for printing essential statistics of the model
        

        Parameters
        ----------
        values : TYPE
            A list with numpy arrays consisting of values, e.g. durations.
        variable_names : TYPE
            The names of the values in value, e.g. 'duration'.
        test : bool
            If True, metrics are from test data. The default is True.
        alpha : TYPE, optional
            The confidence level used for the PI. The default is 0.95.
        n_sf : TYPE, optional
            The number of significant figures (decimals in this case) used in the table. The default is 2.
        results_path : TYPE, optional
            If not None, results are saved to that specified by path. The default is None.

        Returns
        -------
        None.
        """

        # Constructing the headers
        headers = ['variable', 'average', 'std', str(alpha * 100) + '% PI', 'min', 'max']
        rows = []
        n = values[0].shape[0]  # The number of epochs

        # Creating the rows by looping through the values and names
        for value, name in zip(values, variable_names):
            
            # Create the row and add its name
            row = [name]

            # Add the average value
            row.append(round(np.mean(value), n_sf))

            # Add the standard deviation
            row.append(round(np.std(value), n_sf))

            # Add the confidence interval
            CI_size = np.std(value) * t.ppf(1 - (1 - alpha) / 2, n - 1) / np.sqrt(n)  
            CI_size = 0 if np.isnan(CI_size) else CI_size
            # Calculate the CI using the t-distribution
            row.append((round(np.mean(value) - CI_size, n_sf), round(np.mean(value) + CI_size, n_sf)))  # Add the CI

            # Add the minimum value
            row.append(round(np.min(value), n_sf))

            # Add the maximum value
            row.append(round(np.max(value), n_sf))

            # Add the row to the table
            rows.append(row)

        # Print the table
        print("Results: \n")
        print(tabulate(rows, headers=headers))
        
        if results_path:
            fn = "test" if test else "train"
            results_path = results_path if results_path.endswith("/") else results_path + "/"
            results = pd.DataFrame(
                rows,
                columns = headers
            )
            results.to_csv(results_path + fn, index=False)
            print("\nResults saved to " + results_path + fn)

            
    def print_essentials(self, run = 0, test = True, 
                         alpha = 0.95, n_sf = 2, 
                         results_path = None) -> None:
        """Print the essentiel statistics of the model

        Parameters
        ---------
        run : int
            From which run the data should be printed.
        test : boolean
            If statistics on test data should be shown, otherwise train data used
        alpha : float
            The confidence level used for the CI
        n_sf : int
            The number of significant figures (decimals in this case) used in the table
        results_path : string, Optional
            If not None, results are saved to that specified by path
            
        Returns
        -------
        None
        """
        # Select the wanted objects
        if test:
            metrics = self.metrics_test
            roc = self.roc_test
        else:
            metrics = self.metrics_train
            roc = self.roc_train

        # Create a list with the wanted values
        values = [metrics.accuracies[:, run] * 100,
            metrics.durations[:, run],
            metrics.infeasible_percentage[:, run] * 100,
            metrics.infeasible_guessed_percentage[:, run] * 100,
            metrics.feasible_recall[:, run] * 100,
            metrics.feasible_precision[:, run] * 100,
            metrics.infeasible_recall[:, run] * 100,
            metrics.infeasible_precision[:, run] * 100,
            roc.aucs[:, run]]

        # Create a list with the variable names
        variable_names = ['weighted accuracy [%]',
            'duration [s]',
            'infeasible_percentage [%]',
            'infeasible_guessed_percentage [%]',
            'feasible_recall [%]',
            'feasible_precision [%]',
            'infeasible_recall [%]',
            'infeasible_precision [%]',
            'auc of roc']
        
        # Call the helper function
        self.print_essentials_helper(
            values=values, 
            variable_names=variable_names,
            test=test,
            alpha=alpha, 
            n_sf=n_sf, 
            results_path=results_path
        )


    def plot_graphs(self, error_choice = '95PI', 
                    duration = False, accuracies = True, 
                    infeasibility = True, AUC = True) -> None:
        """ Plot results from training.

        Parameters
        ----------
        error_choice : str, optional
            Prediction interval. The default is '95PI'.
        duration : boolean, optional
            Plot training times plot. The default is False.
        accuracies : boolean, optional
            Plot accuracies plot. The default is True.
        infeasibility : boolean, optional
            Plot infeasibility. The default is True.
        AUC : boolean, optional
            Plot Area Under Curve . The default is True.

        Returns
        -------
        None.
        """
        # Plot graphs of the results

        # error_choice      :: which metric that should be used for the error bars: 'std', '95CI', else --> min/max

        # Plotting training times
        if duration:
            print('Training times:')
            plot_durations(self.metrics_test, error_choice = error_choice)

        # Plotting accuracies
        if accuracies:
            print('Weighted accuracies:')
            plot_accuracies(self.metrics_test, self.metrics_train, error_choice = error_choice)

        # Plotting confusion matrices
        print('Confusion matrices:')
        plot_CMs(self.metrics_test, self.metrics_train, error_choice = error_choice)

        if infeasibility:
            # Plotting the percentage of infeasible points in the test and training data
            print('Percentage infeasible on test and train data:')
            plot_infeasibilities(self.metrics_test, self.metrics_train, error_choice = error_choice)

        if AUC:
            # Plotting the AUC on the test and training data
            print('AUC:')
            self.roc_test.plot_auc_combined(self.roc_train, self.iter_lags, error_choice = error_choice)

    
    def hyperparameter_tuning_CV(self, model, n_iter, 
                                 score = 'balanced_accuracy', 
                                 update_parameters = True, 
                                 sampling = 'under', seed = 0, 
                                 **parameters) -> None:
        """Performs a grid search on the model given the user-input lists of parameters

        Parameters
        ----------
        model : callable
            sklearn callable to input to GridSearhCV.
        n_iter : int
            how many iterations of data the model should be evaluated on
        score : string
            which score that should be used to evalute the model, see here https://scikit-learn.org/stable/modules/model_evaluation. The default is 'balanced_accuracy'
        update_parameters : boolean.
            whether or not the parameters given when the object was initialized should be updated. The default is True.
        sampling : string.
            how the data should be balanced, 'under' = undersampling, 'over' = oversampling, else no balancing. The default is 'under'.
        seed : int.
            Seed for randomization. The default is 0.
        **parameters : keyword-parameter pairs
            parameters for the model.

        Returns
        -------
        None.

        """
        # Variables used to select the data. Only n_iter has a large impact on the grid search. start_iter could have a small impact.
        starting_iter = 0
        iter_lag = 0
        n_test = 1

        # Select the data
        X_train, y_train, _, _ = self.select_data(starting_iter, n_iter, iter_lag, n_test, sampling, seed)

        # Initialize the grid search
        gs = GridSearchCV(model(), parameters, scoring = score)

        # Perform the grid search
        gs.fit(X_train, y_train)

        # Print the results
        print("\nThe grid search is complete, here are the results:\n")

        results = gs.cv_results_

        print(pd.DataFrame.from_dict(results))

        print("\nThe best parameters where:\n")

        print(gs.best_params_)

        # Update the parameters
        if update_parameters:
            self.parameters = gs.best_params_


    def hyperparameter_tuning(self, n_train, n_test, 
                              n_epochs = 10, score = 'balanced_accuracy',
                              update_parameters = 'True', sampling = 'under', 
                              seed = 0, progressbar = True, 
                              results_path=None, **parameters) -> dict:
        """Bruteforce testing of all parameter combinations and ranks them.         

        Parameters
        ----------
        n_train : int
            the model that is to be evaluated, e.g. SVC. 
        n_test : int
            The number of iterations included in the training data
        n_epochs : int, optional
            How many times each combination is tested. The default is 10
        score : string, optional
            Which score that should be used to evalute the model. The default is 'balanced_accuracy'.
        update_parameters : boolean, optional
            Hether or not the parameters given when the object was initialized should be updated. The default is 'True'.
        sampling : string
            How the data should be balanced, 'under' = undersampling, 'over' = oversampling, else no balancing. The default is 'under'.
        seed : optional
            Set random seed. The default is 0.
        progressbar : boolean, optional
            If a progressbar should be shown during the training elapses. The default is True.
        results_path : string, optional
            File to save the gridsearch table to. If not None, saves to path. The default is None.
        **parameters : keywords
            Parameters for the model

        Returns
        -------
        best_parameters : dict
            returns the best parameters in the end.

        """
        
        # Calculate the number of combinations
        n = 1
        for list_values in parameters.values():
            n*= len(list_values)

        # Saving old parameter values
        old_parameters = self.parameters
        old_n_epochs, self.default_n_epohcs = self.default_n_epohcs, n_epochs

        # Saving scores
        print_combinations= []
        combinations = []
        values = np.zeros(n)
        stds = np.zeros(n)    
        
        if progressbar == False:
            # print table with tested iteration instead
            print(tabulate(parameters,headers="keys").split("\n")[0])
            print(tabulate(parameters,headers="keys").split("\n")[1])

        for i, params in enumerate(tqdm((list(itertools.product(*parameters.values()))), disable =  not progressbar)):
            # Creates a dictionary with the parameter values that should be tested
            param_combination = dict(zip(list(parameters.keys()), list(params)))
            
            if progressbar == False:
                print(tabulate(parameters,headers="keys").split("\n")[i+2])

            # Updates the parameters to the new combination
            self.parameters = param_combination

            # Trains the model with the new parameters
            self.training_loop(n_train=n_train, n_test=n_test, iter_lags=[0], sampling=sampling, seed=seed)

            # Saves the scores
            if score == 'balanced_accuracy':
                values[i] = np.mean(self.metrics_test.accuracies, axis=0)
                stds[i] = np.std(self.metrics_test.accuracies, axis=0)
            else:
                score = 'AUC'
                values[i] = np.mean(self.roc_test.aucs, axis=0)
                stds[i] = np.std(self.roc_test.aucs, axis=0)

            # Saves the combination
            print_combinations.append(list(param_combination.values()))
            combinations.append(param_combination)
            
        # Finds which combinations performed the best
        order_indeces = np.argsort(-values)
        order = np.argsort(order_indeces) + 1

        # Creates a table that will be printed
        rows = []

        # Only ten best are printed
        for j in range(min(n, 10)):
            rows.append([order[order_indeces[j]]]+print_combinations[order_indeces[j]]+[values[order_indeces[j]], stds[order_indeces[j]]])

        # Prints the tables
        t = tabulate(rows, headers=['Ranking']+list(param_combination.keys())+[score, 'Standard deviation'])
        print("\n", t)    
        if results_path:
            with open(results_path, 'w') as f:
                f.write(t)

        # Extracts the best parameters
        best_parameters = combinations[np.argmax(values)]

        # Updates parameters if that was selected
        if update_parameters:
            self.parameters = best_parameters
        else:
            self.parameters = old_parameters

        self.default_n_epohcs = old_n_epochs

        return best_parameters

        