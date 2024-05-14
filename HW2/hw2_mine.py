# Deric Shaffer
# CS487 - HW2
# Due Date - Feb. 11th, 2024
# Mine Data Main File

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from perceptron import Perceptron
from adaline import AdalineGD
from stochastic import AdalineSGD

# perceptron function
def ppn(x, y):
    # training
    p = Perceptron(eta=0.1, n_iter=10)
    p.fit(x, y)

    # plot training graph
    plt.plot(range(1, len(p.errors_) + 1), p.errors_, marker='o')
    plt.title('Perceptron')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')

    plt.show()

    # plot decision regions
    decision_regions(x, y, classifier=p)
    plt.title('Perceptron')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Height (cm)')
    plt.legend(loc='upper right')

    plt.show()

# adaline function
def ada(x, y):
    ada = AdalineGD(eta=0.5, n_iter=20)

    # standardize features
    x_std = np.copy(x)
    x_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
    x_std[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

    # training
    ada.fit(x_std, y)

    # plot training graph
    plt.plot(range(1, len(ada.losses_) + 1), ada.losses_, marker='o')
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('Epochs')
    plt.ylabel('Mean squared error')

    plt.show()

    # plot decision regions
    decision_regions(x_std, y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('Voltage (standardized)')
    plt.ylabel('Height (standardized)')
    plt.legend(loc='upper right')

    plt.show()

# sgd function
def sgd(x, y):
    sgd = AdalineSGD(eta=0.01, n_iter=15, random_state=1)

    # standardize features
    x_std = np.copy(x)
    x_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
    x_std[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

    # training
    sgd.fit(x_std, y)

    # plot training graph
    plt.plot(range(1, len(sgd.losses_) + 1), sgd.losses_, marker='o')
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('Epochs')
    plt.ylabel('Average loss')

    plt.show()

    # plot decision regions
    decision_regions(x_std, y, classifier=sgd)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('Voltage (standardized)')
    plt.ylabel('Height (standardized)')
    plt.legend(loc='upper right')

    plt.show()


# decision regions function
def decision_regions(x, y, classifier, resolution=0.02):
    # set marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot decision surface
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for i, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1], 
                    alpha=0.8, c=colors[i], marker=markers[i], 
                    label=f'Class {cl}', edgecolors='black')


def main():
    # classifier name options
    classifiers = ['perceptron', 'adaline', 'sgd']

    # get classifier name
    print('Enter a classifier (options: perceptron, adaline, sgd)\n')
    c = input()

    if c not in classifiers:
        print('invalid classifier')

    # get data file
    try:
        data_path = 'mine_data.csv'
        data = pd.read_csv(data_path, header=None)
    except:
        print(f'{data_path} does not exist in local directory')
    
    # get x and y
    y = data.iloc[:, 3].values
    y = np.where(y == '1', 0, 1)

    x = data.iloc[:, [0, 1]].values

    # run classifier on dataset
    if c == classifiers[0]:
        # perceptron
        ppn(x, y)
    elif c == classifiers[1]:
        # adaline
        ada(x, y)
    elif c == classifiers[2]:
        # sgd
        sgd(x, y)



if __name__ == '__main__':
    main()