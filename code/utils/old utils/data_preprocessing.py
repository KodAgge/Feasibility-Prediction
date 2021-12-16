import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from import_data import importData
from import_data import preprocessData

# Import data
file_data = "initial-data.csv"
file_labels = "initial-labels.csv"
n_features = -1

data, labels = importData("initial-data.csv", "initial-labels.csv", n_features)


# PCA fitting, maximum likelihood estimation
if False:
    from sklearn.decomposition import PCA
    pca = PCA(n_components = "mle")

    principalComponents = pca.fit_transform(data.values)
    print("The optimal number of features (MLE) is:", pca.components_.shape[1])

# Pre-process data
X_train, X_test, y_train, y_test = preprocessData(data, labels, test_size = 0.2, seed = 42)

# Random forest test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

clf = RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=None, max_features="sqrt", bootstrap=True, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print()
print("Percentage of infeasible guessed:", sum(y_pred == np.ones(y_pred.shape[0])) / len(y_pred) * 100)
print("Accuracy", sum(y_pred == y_test) / len(y_pred) * 100)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))