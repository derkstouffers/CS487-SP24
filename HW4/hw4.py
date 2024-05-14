# Deric Shaffer
# CS487 - HW4
# Due Date - March 10th. 2024

import time
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_openml

# to fix the ssl verify failed error I keep running into with the mnist dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def pca(x_train, x_test):
    p = PCA(n_components=1, random_state=19)
    #p = PCA(n_components=3, random_state=19)

    start_time = time.time()
    pca_train = p.fit_transform(x_train)
    end_time = time.time()

    print(f'PCA fit time: {end_time - start_time:.4f} seconds')

    pca_test = p.transform(x_test)

    return pca_train, pca_test



def lda(x_train, y, x_test):
    l = LinearDiscriminantAnalysis()
    #l = LinearDiscriminantAnalysis(n_components=2)

    start_time = time.time()
    lda_fit = l.fit(x_train, y)
    end_time = time.time()

    lda_train = lda_fit.transform(x_train)
    lda_test = lda_fit.transform(x_test)

    print(f'LDA fit time: {end_time - start_time:.4f} seconds')

    return lda_train, lda_test





def kpca(x_train, x_test):
    k = KernelPCA()
    #k = KernelPCA(kernel='rbf')

    start_time = time.time()
    kpca_train = k.fit_transform(x_train)
    end_time = time.time()

    kpca_test = k.transform(x_test)

    print(f'Kernel PCA fit time: {end_time - start_time:.4f} seconds')

    return kpca_train, kpca_test



def dt(x_train, x_test, y_train, y_test):
    # training
    d = DecisionTreeClassifier()
    d.fit(x_train, y_train)

    # test
    d_predict = d.predict(x_test)

    # get & print accuracy
    d_accuracy = accuracy_score(y_test, d_predict)
    print(f'Decision Tree Accuracy = {d_accuracy:.4f}')



def main():
    # get user input
    print('\n\tWhich Algorithm Do You Want To Run?')
    print('---------------------------------------------------')
    print('type \'pca\' for Principal Component Analysis')
    print('type \'lda\' for Linear Discriminant Analysis')
    print('type \'kpca\' for Kernel Principal Component Analysis\n')

    # end='' is added so the user input is on the same line (for formatting purposes)
    print('Selected Algorithm: ', end='')
    a = input()

    print('\n\tWhich Dataset Do You Want To Use?')
    print('---------------------------------------------------')
    print('type \'iris\' for the Iris Dataset')
    print('type \'mnist\' for MNIST Dataset\n')

    print('Selected Dataset: ', end='')
    d = input()
    print('\n')

    # error check user input
    algs = ['pca', 'lda', 'kpca']
    data_options = ['iris', 'mnist']

    if d not in data_options:
        print('Unidentified Dataset Entered')
        return
    elif a not in algs:
        print('Unidentified Algorithm Entered')
        return

    # load datasets
    if d == 'iris':
        columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
        iris_data = pd.read_csv('iris.data', names=columns)
    elif d == 'mnist':
        mnist_data = fetch_openml('mnist_784', version=1, cache=True)

    # extract features and target variable
    if d == 'iris':
        x = iris_data.drop('class', axis=1)
        y = iris_data['class']

        # split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=19)

    elif d == 'mnist':
        # get subset of mnist data, first 2000 instances
        x = mnist_data.data[:2000]
        y = mnist_data.target[:2000]

        # split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=19, stratify=y)


    # run algorithms
    if a == 'pca':
        # reduce dimensionality
        reduced_train, reduced_test = pca(x_train, x_test)
    elif a == 'lda':
        # reduce dimensionality
        reduced_train, reduced_test = lda(x_train, y_train, x_test)
    elif a == 'kpca':
        # reduce dimensionality
        reduced_train, reduced_test = kpca(x_train, x_test)

    # use reduced data on decision tree classifier
    dt(reduced_train, reduced_test, y_train, y_test)

if __name__ == '__main__':
    main()