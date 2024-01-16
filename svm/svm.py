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
from sklearn.svm import SVC

if __name__ == '__main__':
    args = sys.argv[1:]
    num_samples = int(args[0])
    num_threads = int(args[1])
    d4p.daalinit(num_threads)
    iterations = int(args[2])
    sgx_flag = bool(args[3])

    X_train, X_test, y_train, y_test = pre.preprocess_svm(num_samples, 4)
    with open("results/result_svm.csv","a") as file:
        for iter in range(iterations):
            file.write('svm,')
            file.write(str(num_samples)+',')
            file.write(str(num_threads)+',')
            file.write(str(sgx_flag)+',')
            clf = SVC(kernel='linear', random_state=42)
            tt.execute(clf, True, X_train, X_test, y_train, y_test, file)