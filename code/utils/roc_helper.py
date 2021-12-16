import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import numpy as np


class ROCObject:


    def __init__(self, n_runs, n_epochs):
        # n_runs        :: number of runs per epoch/seed, e.g. training on [1, 2, 3, 4] iterations means 4 runs
        # n_epochs      :: number of epochs, e.g. starting on iteration [0, 10, 20, 30] means 4 epochs

        self.n_runs = n_runs
        self.n_epochs = n_epochs

        self.i = 0 # Run
        self.j = 0 # Epoch

        self.fp_rates = [[] for i in range(n_runs)]
        self.tp_rates = [[] for i in range(n_runs)]
        self.aucs = np.zeros((n_epochs, n_runs))


    def update_rates(self, y_test, y_score, pos_label = 0):
        # y_test        :: true labels of test set
        # y_score       :: probabilities of positive (feasible) label
        # pos_label     :: value of positive label (0 for us)

        # Calculates curve
        fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label = 0)
        roc_auc = auc(fpr, tpr)

        # Saves values
        self.fp_rates[self.i].append(fpr)
        self.tp_rates[self.i].append(tpr)
        self.aucs[self.j, self.i] = roc_auc

        self.i += 1

        if self.i == self.n_runs: # When all the runs for one epoch is complete
            self.i = 0
            self.j += 1


    def plot_roc(self, epoch_number, run_number):
        # epoch_number  :: from which epoch to plot
        # run_number    :: from which run to plot

        plt.figure()
        lw = 2
        plt.plot(self.fp_rates[epoch_number][run_number], self.tp_rates[epoch_number][run_number], color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % self.aucs[epoch_number][run_number])

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    
    def plot_auc(self, x_values, x_label = 'Iter lag', error_choice = 'std'):
        # x_values      :: the values that should be used on the x-axis

        # Average the auc values
        aucs = average(self.aucs)

        # Get errors and legend
        (errors, legend) = get_errors(self.aucs, 'AUC', error_choice)

        # Plot the errors
        plot_errorbars(x_values, aucs, errors, legend)

        # Show the plot
        show_plot(title = 'Area under curve (AUC)', xlabel = x_label, ylabel = 'AUC', ylim = [0.0, 1.05])


    def plot_auc_combined(self, roc_train, x_values, x_label = 'Iter lag', error_choice = 'std'):
        # roc_train     :: a ROC_helper object for the train data
        # x_values      :: the values that should be used on the x-axis

        # Average the auc values
        aucs_test = average(self.aucs)
        aucs_train = average(roc_train.aucs)

        # Get errors and legend
        (errors_test, legend_test) = get_errors(self.aucs, 'Test AUC', error_choice)
        (errors_train, legend_train) = get_errors(roc_train.aucs, 'Train AUC', error_choice)

        # Plot the errors
        plot_errorbars(x_values, aucs_test, errors_test, legend_test)
        plot_errorbars(x_values, aucs_train, errors_train, legend_train)

        # Show the plot
        show_plot(title = 'Area under curve (AUC)', xlabel = x_label, ylabel = 'AUC', ylim = [0.0, 1.05])


def average(values):
    # Returns the values over epochs
    return np.mean(values, axis = 0)


def get_errors(values, measure, error_choice = 'std'):
    # values        :: values to calculate errors of
    # measure       :: (string) the kind of measure the values are, e.g. accuracy or duration
    # error_choice  :: choice of error, e.g. min/max

    # Returns the errors over epochs and the appriopriate legend

    if error_choice == 'std':
        # Standard deviation
        errors = np.std(values, axis = 0)
        legend = measure + ' w. std bars'

    elif error_choice == '95PI':
        errors = np.std(values, axis = 0) * 1.96 / np.sqrt(values.shape[1])
        legend = measure + ' w. 0.95 PIs'

    else:
        # Min/max
        errors_lower = np.min(values, axis=0) - np.mean(values, axis = 0)
        errors_upper = np.max(values, axis=0) - np.mean(values, axis = 0)
        errors = np.vstack((errors_lower, errors_upper))
        legend = measure + ' w. min/max bars'

    return (errors, legend)


def plot_errorbars(x, y, errors, label):
    # Plotting
    # Simple function so we use the same everywhere
    plt.errorbar(x, y, yerr=errors, fmt= '-o', capsize = 4, capthick = 2, label=label)


def show_plot(title = 'Title', xlabel = 'x', ylabel = 'y', ylim = None):
    # title     :: title used for the plot
    # xlabel    :: label used for x-axis
    # ylabel    :: label used for y-axis
    # ylim      :: (optional) used to limit the y-axis

    # Showing the plot
    if ylim != None:
        plt.ylim(ylim)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True)
    plt.legend()
    plt.show()


def predict_proba(X_test, classifier):
    # X_test        :: testdata
    # classifier    :: classifier object, e.g. SVC or RandomForestClassifier

    # Calculates classification probabilities
    # Needs SVM to have 'probabilities = True'

    probabilities = classifier.predict_proba(X_test)
    probabilities_feasible = probabilities[:, 1]

    return probabilities_feasible