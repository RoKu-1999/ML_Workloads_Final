import os
import sys
import numpy as np
import pandas as pd
import time
import csv
import daal4py as d4p
from sklearnex import patch_sklearn
patch_sklearn()
import sklearn
import preprocess as pre
import train_test as tt
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    args = sys.argv[1:]
    filepath = args[0]
    iterations = int(args[1])
    sgx_flag = int(args[2])
    max_depth = int(args[3])
    if max_depth == 0:
        max_depth = None
    train_size = int(args[4])
    train_size = train_size / 100

    data = pre.read_from_file(filepath)

    X_train, X_test, y_train, y_test = pre.preprocess(data, 1, False, train_size=train_size)
    with open("results/result_dt.csv","a") as file:
        for iter in range(iterations):
            file.write('dt,')
            file.write(str(sgx_flag)+',')
            file.write(str(max_depth)+',')
            file.write(filepath+',')
            file.write(str(train_size)+',')
            clf = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=42)
            tt.execute(clf, False, X_train, X_test, y_train, y_test, file)