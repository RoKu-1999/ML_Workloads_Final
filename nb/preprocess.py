import numpy as np
import pandas as pd
import csv
import math

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.datasets import make_blobs

def read_from_file(filename):
    # print filename
    mylist = []
    for chunk in  pd.read_csv(filename, header=0, chunksize=20000):
        mylist.append(chunk)
    data = pd.concat(mylist, axis= 0)
    del mylist
    return data

def preprocess(data, sampling_factor, entry_sampling=True, train_size=0.7):
    # Filter all numeric data
    col_names = data.columns.values.tolist()
    cols_all = data.columns
    cols_numeric = data._get_numeric_data()
    filtered = list(set(cols_all) - set(cols_numeric))
    col_names = list(filter(lambda a: a not in filtered and a != 'label', col_names))
    # sample data if the classifier is gradient based
    data_entries = data[data.columns[0]].count()
    data = data.sample(n=int(data_entries/sampling_factor),axis='rows',replace=True,random_state=42)
    # take col_names which are all numeric columns as data X
    if entry_sampling == False:
        col_names = col_names[0:int(len(col_names)/sampling_factor)]
    X = data[col_names]
    # label data is y and in column 'label'
    y = data.label
    #Replace all nan with mean strategy (Imputation)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(X)
    X = imputer.transform(X)
    # Encode labels (nothing to be done as all our values are numeric)
    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    # Encode labels (yes, no => 1, 0)
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    # 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, train_size=train_size)
    # Feature Scaling normalizes the input data to a scale
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    # return train, test and labels
    return X_train, X_test, y_train, y_test


def preprocess_svm(n_samples, centers):
    X, y = make_blobs(n_samples=n_samples, centers=centers,
                  random_state=42, cluster_std=0.20)
    return X[0:math.ceil(len(X)*0.7)], X[math.ceil(len(X)*0.7):len(X)], y[0:math.ceil(len(X)*0.7)], y[math.ceil(len(X)*0.7):len(X)]
    
