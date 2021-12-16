"""
Initially, the second raw data file received from project client did not seem correct.
This file tried to find the root of the problem to report back to the client.
"""

from math import ceil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas.core.algorithms import duplicated
from sklearn.model_selection import train_test_split
import csv

n_features = 2
data_dir = "\data\\"
file_data , file_labels = "data.csv", "labels.csv"

cwd = os.getcwd()


# cwd+data_dir+file_data

with open(cwd+data_dir+file_data, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    errors = 0
    count = 0
    error_type = []
    error_count = []
    for row in spamreader:
        count += 1
        length = len(row)
        if length != 3851:
            errors += 1
            print("Error #" + str(errors) + " on row " + str(count) + ": len = " + str(length))
            if length not in error_type:
                error_type.append(length)
                error_count.append(0)
            error_count[error_type.index(length)] += 1

print(error_type)
print(error_count)

            

if False:

    try:
        cwd = os.path.dirname(os.getcwd())

        #labels = pd.read_csv(cwd+data_dir+file_labels,header=None, nrows=1,sep=',',float_precision='high',index_col=False).values.tolist()[0]
        labels = pd.read_csv(cwd+data_dir+file_labels,header=None, sep=',',float_precision='high',index_col=False).values.tolist()[0]
        df_data = pd.read_csv(cwd+data_dir+file_data, header=None,sep=',',index_col=False)
    except FileNotFoundError:
        cwd = os.getcwd()

        # labels = pd.read_csv(cwd+"\data\\"+file_labels,header=None, nrows=1,sep=',',float_precision='high',index_col=False).values.tolist()[0]
        labels = pd.read_csv(cwd+"\data\\"+file_labels,header=None, sep=',',float_precision='high',index_col=False).values.tolist()[0]
        df_data = pd.read_csv(cwd+"\data\\"+file_data, header=None,sep=',',index_col=False)

    df_labels = pd.DataFrame(data={'label':labels,'binary-label':[1 if x!=0 else 0 for x in labels]})

    if n_features != -1:
        df_data = df_data[np.arange(0,min(n_features, df_data.shape[1]),1)]

    print("Number of samples:   ", df_data.shape[0])
    print("Number of features:  ", df_data.shape[1])