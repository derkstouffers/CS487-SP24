# Deric Shaffer
# CS487 - HW1
# Due Date - Jan. 30th, 2024

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# names/titles of columns in dataset
columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']

# Q2
def read_dataset(data):
    print(f'Number of rows = {len(data)}')
    print(f'Number of columns = {len(data.columns)}')

# Q3
def distinct(data):
    # get unique values
    d = data['class'].unique()
    print(d)

# Q4
def setosa(data): 
    # get the rows with class = Iris-setosa
    s = data[data['class'] == 'Iris-setosa']

    print(f'Number of rows = {len(s)}')
    print(f'Average of First Column = {s['sepal length'].mean()}')
    print(f'Max Value of Second Column = {s['sepal width'].max()}')
    print(f'Min Value of Third Column = {s['petal length'].min()}')

# Q5
def plot(data):
    # create scatter plot
    sns.set(style='whitegrid')

    # use seaborn library to show points in different colors and shapes
    sns.scatterplot(x='sepal length', y='sepal width', hue='class', style='class', data=data,
                    palette={'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'},
                    markers={'Iris-setosa': 'o', 'Iris-versicolor': '^', 'Iris-virginica': 's'})

    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.legend()
    plt.title('Iris Data Scatter Plot')

    # display scatter plot
    plt.show()

def main():
    # read in dataset
    iris_data = pd.read_csv('iris.data', names=columns)

    # calculate and print the number of rows and columns in dataset
    read_dataset(iris_data)
    print('\n')

    # get all the values of the last column and print the distinct values of the last column
    distinct(iris_data)
    print('\n')

    # when last column has the value "Iris-setosa", calculate the number of rows, avg of first column, max value of 2nd column, min value of 3rd column
    setosa(iris_data)
    print('\n')

    # draw a scatter plot with the data of the first and second column
    plot(iris_data)

if __name__ == '__main__':
    main()