# Deric Shaffer
# CS487 - HW6
# Due Date - Apr. 9th, 2024

import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml

# to fix the ssl verify failed error I keep running into with the mnist dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def k_elbow(x):
    # use elbow approach to decide reasonable k value
    distortions = []

    for i in range(1, 11):
        k = KMeans(n_clusters=i, init='k-means++', n_init='auto', max_iter=300, random_state=19)
        k.fit(x)
        distortions.append(k.inertia_)

    # plot distortions for different k values
    plt.plot(range(1, 11), distortions, marker='o')
    plt.title('Elbow Approach')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')

    plt.tight_layout()
    plt.show()



# k means for iris dataset
def k_means(x, k):
    km = KMeans(n_clusters=k, init='k-means++', n_init='auto', max_iter=300, random_state=19)
    labels = km.fit_predict(x)

    # plot clusters and centroids
    plt.scatter(x.values[labels == 0, 0], x.values[labels == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Cluster 1')
    plt.scatter(x.values[labels == 1, 0], x.values[labels == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Cluster 2')
    plt.scatter(x.values[labels == 2, 0], x.values[labels == 2, 1], s=50, c='lightblue', marker='v', edgecolor='black', label='Cluster 3')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, marker='*', c='red', edgecolor='black', label='Centroids')

    plt.title('K-Means Clustering')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.show()



def hierarchy(x):
    z = sch.linkage(x, method='average', metric='euclidean')

    # plot dendogram
    plt.figure(figsize=(10,5))
    sch.dendrogram(z)

    plt.title('Hierarchy Clustering Plot')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()



def main():
    # get user input
    print('\tWhich Dataset Do You Want To Use?')
    print('---------------------------------------------------')
    print('\ttype \'iris\' for the Iris Dataset')
    print('\ttype \'mnist\' for MNIST Dataset\n')

    print('Selected Dataset: ', end='')
    d = input()
    print('\n')

    # error check user input
    data_options = ['iris', 'mnist']

    if d not in data_options:
        print('Unidentified Dataset Selected')
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

    elif d == 'mnist':
        # get subset of mnist data, first 2000 instances
        x = mnist_data.data[:2000]
        y = mnist_data.target[:2000]

        # split data into training and testing sets
        x, _, _, _ = train_test_split(x, y, test_size=0.2, random_state=19, stratify=y)

    
    # run elbow approach
    k_elbow(x)

    # get k_val for clustering from user, based on seeing elbow approach graph
    print('What K Value do you want to use based on the graph?\n')
    print('Selected K Value: ', end='')
    k_val = int(input())
    print('\n')

    # run algorithms
    km_start = time.time()
    k_means(x, k_val)
    km_end = time.time()

    sch_start = time.time()
    hierarchy(x)
    sch_end = time.time()

    # algorithm runtimes
    print(f'K-Means Runtime: {km_end - km_start}')
    print(f'Hierarchical Runtime: {sch_end - sch_start}')



if __name__ == '__main__':
    main()