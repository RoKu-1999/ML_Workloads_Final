import time
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

def train(clf, X_train, y_train, file):
    start = time.perf_counter_ns()
    # fit the classifier
    clf = clf.fit(X_train,y_train)
    end = time.perf_counter_ns()
    train_patched = end - start
    file.write(str(train_patched)+',')
    return clf

def predict(clf, X_test, file):
    start = time.perf_counter_ns()
    # predict test set
    y_pred = clf.predict(X_test)
    end = time.perf_counter_ns()
    test_patched = end - start
    file.write(str(test_patched)+',')
    return y_pred

def measure(y_test, y_pred, file):
    # measure accuracy and f1 score
    accuracy = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="weighted")
    file.write(str(accuracy)+',')
    file.write(str(f1)+'\n')

def execute(clf, is_grad, X_train, X_test, y_train, y_test, file):
    file.write(str(X_train.shape[1])+',')
    file.write(str(len(X_train))+',')
    clf = train(clf, X_train, y_train, file)
    file.write(str(len(X_test))+',')
    y_pred = predict(clf, X_test, file)
    measure(y_test, y_pred, file)