import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import t

class MetricObject:

    def __init__(self, n_runs, n_epochs):
        """Metric object         

        Parameters
        ----------
        n_runs : int
            Number of runs per epoch/seed, e.g. training on [1, 2, 3, 4] iterations means 4 runs.
        n_epochs : int
            Number of epochs, e.g. starting on iteration [0, 10, 20, 30] means 4 epochs.

        Returns
        -------
        None.

        """
        
        self.n_runs = n_runs
        self.n_epochs = n_epochs

        # Lists containing metric information
        self.accuracies = np.zeros((n_epochs, n_runs))

        self.x_values = np.zeros((n_epochs, n_runs))

        self.durations = np.zeros((n_epochs, n_runs))

        self.infeasible_guessed_percentage = np.zeros((n_epochs, n_runs))
        self.infeasible_percentage = np.zeros((n_epochs, n_runs))

        self.feasible_recall = np.zeros((n_epochs, n_runs))
        self.feasible_precision = np.zeros((n_epochs, n_runs))
        self.infeasible_recall = np.zeros((n_epochs, n_runs))
        self.infeasible_precision = np.zeros((n_epochs, n_runs))

        self.i = 0 # Keeping track of the current run
        self.j = 0 # Keeping track of the current epoch


    def update_metrics(self, y, y_pred, x_value, duration, printing = False):
        """
        

        Parameters
        ----------
        y : list
            True labels.
        y_pred : list
            Predicted labels.
        x_value : pandas.DataFrame
            x values used for the graph, e.g. #iterations or training size.
        duration : int
            Training time for the run.
        printing : bool, optional
            If True, logs should be printed. The default is False.

        Returns
        -------
        None.

        """ 
        # Update the metrics
        self.accuracies[self.j, self.i] = accuracy_score(y, y_pred)

        self.x_values[self.j, self.i] = x_value

        self.durations[self.j, self.i] = duration

        infeasible_guessed = sum(y_pred == np.ones(y_pred.shape)) / y_pred.shape[0]
        self.infeasible_guessed_percentage[self.j, self.i] = infeasible_guessed

        infeasible = sum(y == np.ones(y.shape)) / y.shape[0]
        self.infeasible_percentage[self.j, self.i] = infeasible

        cm = confusion_matrix(y, y_pred)

        self.feasible_recall[self.j, self.i] = cm[0, 0] / max(sum(cm[0, :]), 1)
        self.feasible_precision[self.j, self.i] = cm[0, 0] / max(sum(cm[:, 0]), 1)

        self.infeasible_recall[self.j, self.i] = cm[1, 1] / max(sum(cm[1, :]), 1)
        self.infeasible_precision[self.j, self.i] = cm[1, 1] / max(sum(cm[:, 1]), 1)

        self.i += 1

        if self.i == self.n_runs: # When all the runs for one epoch is complete
            self.i = 0
            self.j += 1


def accuracy_score(y_true, y_pred):
    """ Weighted accuracy"""
    return balanced_accuracy_score(y_true, y_pred)


def average(values):
    """Returns the values over epochs"""
    return np.mean(values, axis = 0)


def get_errors(values, measure, error_choice = 'std'):
    """ Retrives the errors.
    
    Parameters
    ----------
    values : list
        Values to calculate errors of. No default.
    measure : str
        The kind of measure the values are, e.g. accuracy or duration. No default.
    error_choice : str
        Choice of error, e.g. min/max. The default is 'std'.
        
    Returns
    -------
    errors : list
        calculated errors
    legend : str
        Legend to use for plot.
    """
    if error_choice == 'std':
        # Standard deviation
        errors = np.std(values, axis = 0)
        legend = measure + ' w. std bars'

    elif error_choice == '95PI':
        n = values.shape[0]
        errors = np.std(values, axis = 0) * t.ppf(0.975, n - 1) / np.sqrt(n)
        legend = measure + ' w. 0.95 PIs'

    else:
        # Min/max
        errors_lower = np.min(values, axis=0) - np.mean(values, axis = 0)
        errors_upper = np.max(values, axis=0) - np.mean(values, axis = 0)
        errors = np.vstack((errors_lower, errors_upper))
        legend = measure + ' w. min/max bars'

    return (errors, legend)


def plot_errorbars(x, y, errors, label):
    """Plotting
    
    Simple function so we use the same everywhere.
    """
    plt.errorbar(x, y, yerr=errors, fmt= '-o', capsize = 4, capthick = 2, label=label)


def show_plot(title = 'Title', xlabel = 'x', ylabel = 'y', ylim = None):
    """Prints the plot
    
    Parameters
    ----------
    title : string
        Title used for the plot. Default is 'Title'.
    label : string
        Label used for x-axis. Default is 'x'.
    ylabel : string
        Label used for y-axis. Default is 'y'.
    ylim : list, optional
        Used to limit the y-axis. Default is None
        
    Returns
    -------
    None.
    """
    # Showing the plot
    if ylim != None:
        plt.ylim(ylim)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_durations(M, error_choice = 'std', title = 'Training + prediction times', ylabel = 'Duration [s]', xlabel = 'Iterations forward'):
    """Plotting durations

    Parameters
    ----------
    M : MetricObject
        Instance of MetricObject. No default.
    error_choice : string, optional
        Choice of errors bars. The default is 'std'.
    title : string, optional
        The default is 'Training + prediction times'.
    ylabel : string, optional
        The default is 'Duration [s]'.
    xlabel : string, optional
        The default is 'Iterations forward'.

    Returns
    -------
    None.
    """
    # Get average values
    durations = average(M.durations)
    x_values = average(M.x_values)

    # Get errors and legend
    (errors, legend) = get_errors(M.durations, ylabel, error_choice)

    # Plot the errors
    plot_errorbars(x_values, durations, errors, legend)

    # Show the plot
    show_plot(title, xlabel, ylabel)



def plot_accuracies(M_test, M_train, error_choice = 'std', title = 'Accuracies', ylabel = 'Accuracy %', xlabel = 'Iterations forward'):
    """Plots accuracies on test and training data.

    Parameters
    ----------
    M_test : MetricObject
        MetricObject for results on testdata. No default.
    M_train : TYPE
        MetricObject for results on traindata. No default.
    error_choice : str, optional
        Choice of errors bars. The default is 'std'.
    title : str, optional
        Title of plot. The default is 'Accuracies'.
    ylabel : str, optional
        Label for y-axis. The default is 'Accuracy %'.
    xlabel : str, optional
        Label of x-axis. The default is 'Iterations forward'.

    Returns
    -------
    None.

    """
    # Get averages
    accuracies_test = average(M_test.accuracies)
    accuracies_train = average(M_train.accuracies)
    x_values = average(M_test.x_values)

    # Get errors and legend
    (errors_test, legend_test) = get_errors(M_test.accuracies, 'Test accuracy', error_choice)
    (errors_train, legend_train) = get_errors(M_train.accuracies, 'Train accuracy', error_choice)

    # Plot error graphs
    plot_errorbars(x_values, accuracies_test, errors_test, legend_test)
    plot_errorbars(x_values, accuracies_train, errors_train, legend_train)

    # Show the graph
    show_plot(title, xlabel, ylabel, ylim=[0, 1.1])

    print("Test accuracies:", accuracies_test)


def plot_CM_helper(x, y_test, y_train, indeces, label, axs, error_choice = 'std', label_1 = 'Test', label_2 = 'Train', CM = True):
    """Helper function for plot_CMs.
    
    Parameters
    ----------
    x : array
        x-values.
    y_test : array
        y values for test.
    y_train : array
        y values for train.
    indeces : array
        Where in (2 x 2) matrix it should be plotted.
    label : str
        Used for title and legends.
    axs : pyplot axis object.
        Axis.
    error_choice : str, optional
        choice of errors bars. The default is 'std'.
    label_1 : str, optional
        The default is 'Test'.
    label_2 : str, optional
        The default is 'Train'.
    CM : bool, optional
        The default is True.

    Returns
    -------
    None.

    """
    # Get errors
    (errors_test, _) = get_errors(y_test, '', error_choice)
    (errors_train, _) = get_errors(y_train, '', error_choice)

    # Get averages
    x = average(x)
    y_test = average(y_test)
    y_train = average(y_train)

    # Set title
    axs[indeces].set_title(label.capitalize())

    # Add explanation of errors 
    if error_choice == 'std':
        label += ' w. std bars'
    elif error_choice == '95PI':
        label += ' w. 0.95 PIs'
    else:
        label += ' w. min/max bars'

    label_test = label_1 + ' ' + label
    label_train = label_2 + ' ' + label

    # Plotting
    axs[indeces].errorbar(x, y_test, errors_test, fmt= '-o', capsize = 4, capthick = 2, label = label_test)
    axs[indeces].errorbar(x, y_train, errors_train, fmt= '-o', capsize = 4, capthick = 2, label = label_train)

    axs[indeces].set_ylim([0, 1.1])
    axs[indeces].grid(True)
    axs[indeces].set_xlabel('Iterations forward')
    axs[indeces].legend()


def plot_CMs(M_test, M_train, error_choice = 'std'):
    """Plottings confusion matrices.

    Parameters
    ----------
    M_test : MetricObject
        For results on testdata
    M_train : MetricObject
        For results on traindata
    error_choice : TYPE, optional
        Choice of error bars. The default is 'std'.

    Returns
    -------
    None
    """
    # Plots recall and precision for in-/feasibel on on test and training data
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(12)
    fig.set_figwidth(14)

    plot_CM_helper(M_test.x_values, M_test.feasible_precision, M_train.feasible_precision, (0, 0), 'feasible precision', axs, error_choice)

    plot_CM_helper(M_test.x_values, M_test.feasible_recall, M_train.feasible_recall, (0, 1), 'feasible recall', axs, error_choice)

    plot_CM_helper(M_test.x_values, M_test.infeasible_precision, M_train.infeasible_precision, (1, 0), 'infeasible precision', axs, error_choice)

    plot_CM_helper(M_test.x_values, M_test.infeasible_recall, M_train.infeasible_recall, (1, 1), 'infeasible recall', axs, error_choice)

    plt.show()


def plot_infeasibilities(M_test, M_train, error_choice = 'std', title = 'Infeasibility percentage', ylabel = 'Infeasibility %', xlabel = 'Iterations forward'):   
    """Plotting infeasibilities.

    Parameters
    ----------
    M_test : MetricObject
        For results on testdata.
    M_train : MetricObject
        For results on traindata.
    error_choice : str, optional
        Choice of errors bars. The default is 'std'.
    title : str, optional
        The default is 'Infeasibility percentage'.
    ylabel : str, optional
        The default is 'Infeasibility %'.
    xlabel : str, optional
        The default is 'Iterations forward'.

    Returns
    -------
    None.

    """
    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(4.5)
    fig.set_figwidth(14)

    plot_CM_helper(M_test.x_values, M_test.infeasible_guessed_percentage, M_test.infeasible_percentage, 0, 'infeasibility (test)', axs, error_choice, 'Guessed', 'Actual')

    plot_CM_helper(M_test.x_values, M_train.infeasible_guessed_percentage, M_train.infeasible_percentage, 1, 'infeasibility (train)', axs, error_choice, 'Guessed', 'Actual')

    plt.show()


def plot_infeasibility(M, error_choice = 'std', title = 'Infeasibility percentage', ylabel = 'Infeasibility %', xlabel = 'Iterations forward'):
    """Plotting infeasibility.

    Parameters
    ----------
    M : MetricObject
        MetricObject
    error_choice : str, optional
        Choice of error bars, 95PI. The default is 'std'.
    title : string, optional
        The default is 'Infeasibility percentage'.
    ylabel : string, optional
        The default is 'Infeasibility %'.
    xlabel : string, optional
        The default is 'Iterations forward'.

    Returns
    -------
    None.

    """
    # Get averages
    infeasible_guessed_percentage = average(M.infeasible_guessed_percentage)
    infeasible_percentage = average(M.infeasible_percentage)
    x_values = average(M.x_values)

    # Get errors and labels
    (errors_guessed, legend_guessed) = get_errors(M.infeasible_guessed_percentage, 'Infeasible guessed', error_choice)
    (errors_actual, legend_actual) = get_errors(M.infeasible_percentage, 'Actual infeasible', error_choice)

    # Plot errors graphs
    plot_errorbars(x_values, infeasible_guessed_percentage, errors_guessed, legend_guessed)
    plot_errorbars(x_values, infeasible_percentage, errors_actual, legend_actual)

    # Show the plot
    show_plot(title, xlabel, ylabel, ylim=[0, 1.1])