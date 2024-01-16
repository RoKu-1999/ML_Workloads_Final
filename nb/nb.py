import os
import sys
import numpy as np
import pandas as pd
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
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':
    args = sys.argv[1:]
    filepath = args[0]
    iterations = int(args[1])
    sgx_flag = int(args[2])

    data = pre.read_from_file(filepath)

    X_train, X_test, y_train, y_test = pre.preprocess(data, 1, False)
    with open("results/result_nb.csv","a") as file:
        for iter in range(iterations):
            file.write('nb,')
            clf = GaussianNB()
            file.write(str(sgx_flag)+',')
            tt.execute(clf, False, X_train, X_test, y_train, y_test, file)
