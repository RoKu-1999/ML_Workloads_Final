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
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    args = sys.argv[1:]
    filepath = args[0]
    num_threads = int(args[1])
    d4p.daalinit(num_threads)
    iterations = int(args[2])
    sgx_flag = int(args[3])
    max_depth = int(args[4])
    if max_depth == 0:
        max_depth = None
    n_estimators = int(args[5])
    train_size = int(args[6])
    train_size = train_size / 100
    bootstrap = bool(args[7])

    data = pre.read_from_file(filepath)

    X_train, X_test, y_train, y_test = pre.preprocess(data, 1, False, train_size)
    with open("results/result_rf.csv","a") as file:
        for iter in range(iterations):
            file.write('rf,')
            file.write(str(num_threads)+',')
            file.write(str(sgx_flag)+',')
            file.write(str(max_depth)+',')
            file.write(str(n_estimators)+',')
            file.write(str(train_size)+',')
            clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=num_threads, max_depth=max_depth, bootstrap=bootstrap)
            tt.execute(clf, False, X_train, X_test, y_train, y_test, file)
