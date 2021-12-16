from math import ceil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas.core.algorithms import duplicated
from sklearn.model_selection import train_test_split
import random
from sklearn import preprocessing

def importData(file_data, file_labels, 
               data_dir = "\Code\\", n_features = -1, 
               scaled = False, chunks = False, 
               n_datapoints = -1, print_log=True, 
               iterations=None,reduced=False):
    """Reads and prepares the data.
    
    Parameters
    ----------
    file_data : str
        name of data file
    data_dir : str
        directory for data
    file_labels : str
        name of labels file
    n_features : int
        number of features to be included
    print_log : boolean
        if logs should be printed when run
    iterations : int
        number of samples in each iteration to label in file_labels
    reduced : boolean
        if True, only a subset of the csv will be loaded
        
    Returns
    -------
    df_data : pd.DataFrame
        features
    df_labels:  pd.DataFrame
        targets
    """
    readf = lambda f: (
        pd.read_feather(cwd+data_dir+f) if file_data.endswith(".feather") 
        else
        pd.read_csv(cwd+data_dir+f, header=None,sep=',',index_col=False,dtype='float64',nrows=nrows)
    )
    try:
        cwd = os.path.dirname(os.getcwd())
        if reduced:
            nrows=1024*20 # 20 iterations probabaly
        else:
            nrows=None
        labels = pd.read_csv(cwd+data_dir+file_labels,header=None,float_precision='high',index_col=False,nrows=nrows)
        df_data = readf(file_data)
                    
    except FileNotFoundError:
        cwd = os.getcwd()
        labels = pd.read_csv(cwd+"\data\\"+file_labels,header=None,float_precision='high',index_col=False,nrows=nrows)
        df_data = readf(file_data)

    df_labels = labels.rename(columns={0:'label'})

    labels[labels != 0] = 1 # Binary labels
    df_labels['binary-label'] = labels.astype(int)

    if chunks != False:
        # chunks=5 if not else specified
        chunks = 5 if chunks==True else chunks
        df_labels['ordinal'] = pd.qcut(df_labels['label'], chunks, labels = False, duplicates='drop')

    if n_features != -1:
        df_data = df_data[np.arange(0,min(n_features, df_data.shape[1]),1)]

    if n_datapoints > 0:
        df_data = df_data[0:min(n_datapoints, df_data.shape[0])]
        df_labels = df_labels[0:min(n_datapoints, df_labels.shape[0])]

    if scaled:
        scaler = preprocessing.StandardScaler()
        standard_df = scaler.fit_transform(df_data)
        df_data = pd.DataFrame(standard_df)
        
    if iterations==None:
        if reduced:
            iterations=20 #
        elif file_data=="data2.csv" or file_data=="data1.csv":
            iterations=100
        else:
            print("Iterations labels could not be infered from file name. Use iterations=False to avoid this message. Or input no. of samples in each generation /Eric")
    if iterations!=False:
        df_labels['iteration']=[x for x in range(0,iterations) for i in range(0,int(df_data.shape[0]/iterations))]
    
    if print_log:
        print("Number of samples:   ", df_data.shape[0])
        print("Number of features:  ", df_data.shape[1])

    return df_data, df_labels


def preprocessData(data, labels, test_size = 0.2, seed = 42, printing = True):
    """Preprocessing, i.e. splitting.
    
    Parameters
    ----------
    data : pd.DataFrame
        Feature set. No default.
    labels : pd.DataFrame
        Target labels. No default.
    test_size : float
        Proportion for the test sample. The default is 0.2.
    seed : int
         Randomizer. Defaults to 42.
    printing : boolean
        If True, logs should be printed when run. The default is True.

    Returns
    -------
    X_train : np.ndarray
        Training feature set.
    X_test :  np.ndarray
        Testing feature set.
    y_train : np.ndarray
        Training target labels
    y_test : np.ndarray
        Testing target labels
    """

    X = data.values
    y = labels['binary-label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = seed)

    if printing:
        print()
        print("Number of train samples:     ", X_train.shape[0])
        print("Number of test samples:      ", X_test.shape[0])

    return X_train, X_test, y_train, y_test


def selectIterations(data, labels, iter_start, n_iter_train, iter_lag, n_iter_test, printing = True):
    """Chooses iterations
    
    Parameters
    ----------
    data : pd.DataFrame
        Feature set. No default.
    labels : pd.DataFrame
        Target labels. No default.
    iter_start : float
        From which iteration to start selecting training data from. No default.
    n_iter_train : int
        How many iterations that should be chosen for the training data. No default.
    iter_lag : boolean
        How many iterations that should be skipped after the traindata when selecting the testdata (= 0 means the iteration just after). No default
    n_iter_test : int
        How many iterations that should be chosen for the testdata. No default.
    printing : bool
        If True, logs should be printed. Defaults to True.
       
    Returns
    -------
    X_train : np.ndarray
        Training feature set.
    X_test :  np.ndarray
        Testing feature set.
    y_train : np.ndarray
        Training target labels
    y_test : np.ndarray
        Testing target labels
    """
    iter_size = 1024

    if (iter_start + n_iter_train + iter_lag + n_iter_test) * iter_size > data.shape[0]:
        raise Exception("Invalid choice: (starting iteration + number of iterations + space + number of test iterations) is larger than size of sample.")

    else:
        train_data = data[(iter_start * iter_size):((iter_start + n_iter_train) * iter_size)]
        train_labels = labels[(iter_start * iter_size):((iter_start + n_iter_train) * iter_size)]

        test_data = data[((iter_start + n_iter_train + iter_lag) * iter_size):((iter_start + n_iter_train + iter_lag + n_iter_test) * iter_size)]
        test_labels = labels[((iter_start + n_iter_train + iter_lag) * iter_size):((iter_start + n_iter_train + iter_lag + n_iter_test) * iter_size)]

        X_train = train_data.values
        y_train = train_labels['binary-label'].values

        X_test = test_data.values
        y_test = test_labels['binary-label'].values

        if printing:
            print()
            print("Number of train samples:     ", X_train.shape[0])
            print("Number of test samples:      ", X_test.shape[0])

        return X_train, y_train, X_test, y_test


def underSampleDataFrame(data, labels, seed = 42):
    """Undersampling of pandas DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        Feature set. No default.
    labels : pd.DataFrame
        Target labels. No default.
    seed : int
        Randomizer. The default is 42.
 
    Returns
    -------
    data_undersampled : pd.DataFrame
        Data with some rows resampled.
    labels_undersampled : pd.DataFrame
        Labels with some resampled.
    """
    random.seed(seed)

    infeasible_indices = labels['binary-label'].values == 1
    feasible_indices = labels['binary-label'].values == 0

    n_infeasible = sum(infeasible_indices)
    n_feasible = sum(feasible_indices)

    data_infeasible = data[infeasible_indices]
    data_feasible = data[feasible_indices]

    labels_infeasible = labels[infeasible_indices]
    labels_feasible = labels[feasible_indices]

    indices = random.sample(range(n_infeasible), n_feasible)

    data_undersampled = pd.concat([data_feasible, data_infeasible.iloc[indices,:]])
    labels_undersampled = pd.concat([labels_feasible, labels_infeasible.iloc[indices,:]])

    print()
    print("Undersampling complete (", data.shape[0], "-->", data_undersampled.shape[0], ") - a reduction of", "{0:.0%}".format((1 - data_undersampled.shape[0] / data.shape[0])))

    return data_undersampled, labels_undersampled


def underSampleNumpyArray(data, labels, seed = 42, printing = True):
    """Undersampling of pandas DataFrame.
    
    Removes some rows corresponding to the majority class in
    order to balance the labels.
    
    Parameters
    ----------
    data : pd.DataFrame
        Feature set. No default.
    labels : pd.DataFrame
        Target labels. No default.
    seed : int
        Randomizer. The default is 42.
    printing : bool
        If True, logs are printed. The default is True.
    
    Returns
    -------
    data_undersampled : pd.DataFrame
        Data with some rows resampled.
    labels_undersampled : pd.DataFrame
        Labels with some resampled.
    """

    random.seed(seed)

    infeasible_indices = labels == 1
    feasible_indices = labels == 0

    n_infeasible = sum(infeasible_indices)
    n_feasible = sum(feasible_indices)

    data_infeasible = data[infeasible_indices, :]
    data_feasible = data[feasible_indices, :]

    labels_infeasible = labels[infeasible_indices]
    labels_feasible = labels[feasible_indices]

    indices = random.sample(range(n_infeasible), n_feasible)

    data_undersampled = np.concatenate((data_feasible, data_infeasible[indices , :]), axis = 0)
    labels_undersampled = np.concatenate((labels_feasible, labels_infeasible[indices]), axis = 0)

    if printing:
        print()
        print("Undersampling complete (", data.shape[0], "-->", data_undersampled.shape[0], ") - a reduction of", "{0:.0%}".format((1 - data_undersampled.shape[0] / data.shape[0])))

    return data_undersampled, labels_undersampled


def overSampleDataFrame(data, labels, seed = 42):
    """Oversampling of pandas DataFrame.
    
    Duplicates some rows corresponding to minority class
    in order to balance the labels.
    
    Parameters
    ----------
    data : pd.DataFrame
        Feature set. No default.
    labels : pd.DataFrame
        Target labels. No default.
    seed : int
        Randomizer. The default is 42.
 
    Returns
    -------
    data_undersampled : pd.DataFrame
        Data with some rows resampled.
    labels_undersampled : pd.DataFrame
        Labels with some resampled.
    """
    random.seed(seed)

    infeasible_indices = labels['binary-label'].values == 1
    feasible_indices = labels['binary-label'].values == 0

    n_infeasible = sum(infeasible_indices)
    n_feasible = sum(feasible_indices)

    data_infeasible = data[infeasible_indices]
    data_feasible = data[feasible_indices]

    labels_infeasible = labels[infeasible_indices]
    labels_feasible = labels[feasible_indices]

    n_copies = int(n_infeasible / n_feasible) + 1

    data_feasible_duplicated = pd.concat([data_feasible for i in range(n_copies)])
    labels_feasible_duplicated = pd.concat([labels_feasible for i in range(n_copies)])
    
    indices = random.sample(range(n_feasible * n_copies), n_infeasible)

    data_oversampled = pd.concat([data_feasible_duplicated.iloc[indices,:], data_infeasible])
    labels_oversampled = pd.concat([labels_feasible_duplicated.iloc[indices,:], labels_infeasible])

    print()
    print("Oversampling complete (", data.shape[0], "-->", data_oversampled.shape[0], ") - an increase of", "{0:.0%}".format((data_oversampled.shape[0] / data.shape[0] - 1)))

    return data_oversampled, labels_oversampled


def overSampleNumpyArray(data, labels, seed = 42):
    """Oversampling of pandas DataFrame.
    
    Duplicates some rows corresponding to minority class
    in order to balance the labels.
    
    Parameters
    ----------
    data : numpy.ndarray
        Feature set. No default.
    labels : numpy.ndarray
        Target labels. No default.
    seed : int
        Randomizer. The default is 42.
 
    Returns
    -------
    data_undersampled : numpy.ndarray
        Data with some rows resampled.
    labels_undersampled : numpy.ndarray
        Labels with some resampled.
    """

    random.seed(seed)
    
    infeasible_indices = labels == 1
    feasible_indices = labels == 0

    n_infeasible = sum(infeasible_indices)
    n_feasible = sum(feasible_indices)

    data_infeasible = data[infeasible_indices, :]
    data_feasible = data[feasible_indices, :]

    labels_infeasible = labels[infeasible_indices]
    labels_feasible = labels[feasible_indices]

    n_copies = int(n_infeasible / n_feasible) + 1

    data_feasible_duplicated = np.concatenate([data_feasible for i in range(n_copies)], axis = 0)
    labels_feasible_duplicated = np.concatenate([labels_feasible for i in range(n_copies)], axis = 0)
    
    indices = random.sample(range(n_feasible * n_copies), n_infeasible)

    data_oversampled = np.concatenate([data_feasible_duplicated[indices,:], data_infeasible])
    labels_oversampled = np.concatenate([labels_feasible_duplicated[indices], labels_infeasible])

    print()
    print("Oversampling complete (", data.shape[0], "-->", data_oversampled.shape[0], ") - an increase of", "{0:.0%}".format((data_oversampled.shape[0] / data.shape[0] - 1)))

    return data_oversampled, labels_oversampled


if __name__ == "__main__":
    # data, labels = importData("initial-data.csv", "initial-labels.csv", data_dir = "\Code\\", n_features = 2)

    data, labels = importData("data.csv", "labels.csv", data_dir = "\\", n_features = -1)

    # data_undersampled, labels_undersampled = underSampleDataFrame(data, labels)

    # data_oversampled, labels_oversampled = overSampleDataFrame(data, labels)

    # X_train, X_test, y_train, y_test = preprocessData(data, labels, test_size = 0.2, seed = 42)

    # X_train_undersampled, y_train_undersampled = underSampleNumpyArray(X_train, y_train)

    # X_train_oversampled, y_train_oversampled = overSampleNumpyArray(X_train, y_train)
