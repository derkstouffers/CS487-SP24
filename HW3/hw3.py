# Deric Shaffer
# CS487 - HW3
# Due Date - Feb. 25th, 2024

import time
import pandas as pd
import numpy as np

from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_digits



# classifier definitions, commented definitions are the "one tuned hyper-parameter" versions
ppn = Perceptron(max_iter=100, random_state=19)
#ppn = Perceptron(max_iter=100, eta0=0.01, random_state=19)

lr = LogisticRegression(max_iter=200, random_state=19)
#lr = LogisticRegression(max_iter=200, C=0.1, random_state=19)

svm = SVC(kernel='linear', random_state=19)
#svm = SVC(kernel='linear', C=0.01, random_state=19)

rbf = SVC(kernel='rbf', random_state=19)
#rbf = SVC(kernel='rbf', C=0.01, random_state=19)

dt = DecisionTreeClassifier(random_state=19)
#dt = DecisionTreeClassifier(criterion='entropy', random_state=19)

knn = KNeighborsClassifier()
#knn = KNeighborsClassifier(n_neighbors = 10)



# training funcion
def training(classifier, x_train, y_train):
    if classifier == 'ppn':
        ppn.fit(x_train, y_train)

    elif classifier == 'lr':
        lr.fit(x_train, y_train)

    elif classifier == 'lsvm':
        svm.fit(x_train, y_train)

    elif classifier == 'nsvm':
        rbf.fit(x_train, y_train)

    elif classifier == 'dt':
        dt.fit(x_train, y_train)

    elif classifier == 'knn':
        knn.fit(x_train, y_train)



# testing function
def testing(classifier, x_test, y_test):
    if classifier == 'ppn':
        ppn_predict = ppn.predict(x_test)

        # get & print accuracy
        ppn_accuracy = accuracy_score(y_test, ppn_predict)
        print(f'Perceptron Accuracy = {ppn_accuracy:.4f}')

    elif classifier == 'lr':
        lr_predict = lr.predict(x_test)

        # get & print accuracy
        lr_accuracy = accuracy_score(y_test, lr_predict)
        print(f'Logisitc Regression Accuracy = {lr_accuracy:.4f}')

    elif classifier == 'lsvm':
        svm_predict = svm.predict(x_test)

        # get & print accuracy
        svm_accuracy = accuracy_score(y_test, svm_predict)
        print(f'Linear SVM Accuracy = {svm_accuracy:.4f}')

    elif classifier == 'nsvm':
        rbf_predict = rbf.predict(x_test)

        # get & print accuracy
        rbf_accuracy = accuracy_score(y_test, rbf_predict)
        print(f'Non-linear SVM Accuracy {rbf_accuracy:.4f}')

    elif classifier == 'dt':
        dt_predict = dt.predict(x_test)

        # get & print accuracy
        dt_accuracy = accuracy_score(y_test, dt_predict)
        print(f'Decision Tree Accuracy = {dt_accuracy:.4f}')

    elif classifier == 'knn':
        knn_predict = knn.predict(x_test)

        # get & print accuracy
        knn_accuracy = accuracy_score(y_test, knn_predict)
        print(f'K-Nearest Neighbors Accuracy = {knn_accuracy:.4f}')



def main():
    # load digits dataset
    digits = load_digits()

    # extract data & labels from dataset
    x = digits.data
    y = digits.target

    # split dataset into training and test sets 80/20
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=19)

    # Get user terminal input
    print('\n Which Classifier do you want to use?')
    print('--------------------------------------')
    print('type \'ppn\' for Perceptron')
    print('type \'lr\' for Logistic Regression')
    print('type \'lsvm\' for Linear SVM')
    print('type \'nsvm\' for Non-linear SVM')
    print('type \'dt\' for Decision Tree')
    print('type \'knn\' for K-Nearest Neighbors\n')

    # end='' is added so the user input is on the same line (for formatting purposes)
    print('Selected Classifier: ', end='')
    c = input()
    print('\n')

    # error checking user input
    c_options = ['ppn', 'lr', 'lsvm', 'nsvm', 'dt', 'knn']

    if c not in c_options:
        print('Classifier not a valid option')
    else:
        # get training time
        start_time_training = time.time()
        training(c, x_train, y_train)
        end_time_training = time.time()

        # get testing time & print out accuracy
        start_time_testing = time.time()
        testing(c, x_test, y_test)
        end_time_testing = time.time()

        # print out training and testing times
        print(f'Training Time = {(end_time_training - start_time_training):.4f} seconds')
        print(f'Testing Time = {(end_time_testing - start_time_testing):.4f} seconds\n')


if __name__ == '__main__':
    main()