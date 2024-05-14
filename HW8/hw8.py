# Deric Shaffer
# CS487 - HW8
# Due Date - May 5th, 2024

import time

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# dataset imports
from sklearn.datasets import load_digits
from ucimlrepo import fetch_ucirepo

def bagging(x_train, y_train, x_test, y_test):
    # base classifier
    #base = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=19)
    base = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=19)

    # bagging classifier
    #b = BaggingClassifier(estimator=base, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=19)
    b = BaggingClassifier(estimator=base, n_estimators=250, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=19)

    # get training runtime
    base_start = time.time()
    tree = base.fit(x_train, y_train)
    base_end = time.time()

    tree_pred = tree.predict(x_test)
    tree_test = accuracy_score(y_test, tree_pred)

    # get training runtime
    bag_start = time.time()
    bag = b.fit(x_train, y_train)
    bag_end = time.time()

    bag_pred = bag.predict(x_test)
    bag_test = accuracy_score(y_test, bag_pred)

    return (base_end - base_start), (bag_end - bag_start), tree_test, bag_test



# random forest classifier
def random_forest(x_train, y_train, x_test, y_test):
    #forest = RandomForestClassifier(criterion='gini', n_estimators=25, n_jobs=2, random_state=19)
    forest = RandomForestClassifier(criterion='entropy', n_estimators=25, n_jobs=2, random_state=19)

    # get training runtime
    forest_start = time.time()
    forest.fit(x_train, y_train)
    forest_end = time.time()

    forest_pred = forest.predict(x_test)
    forest_test = accuracy_score(y_test, forest_pred)

    return (forest_end - forest_start), forest_test



def main():
    # get user input
    print('\nWhich dataset do you want to run the classifiers on?')
    print('-----------------------------------------------------')
    print('Type \'1\' to use the Digits Dataset')
    print('Type \'2\' to use the Mammographic Mass Dataset')

    print('\nSelected Dataset: ', end='')
    user_choice = input()
    print()

    # error check user input
    if user_choice not in ['1', '2']:
        print('Unidentified Dataset Selected')
        return

    # fetch selected dataset
    if user_choice == '1':
        digits = load_digits()

        data = digits.data
        target = digits.target

    elif user_choice == '2':
        mammographic_mass = fetch_ucirepo(id=161)
     
        data = mammographic_mass.data.features 
        target = mammographic_mass.data.targets

        # deal with missing values
        imputer = SimpleImputer(strategy='median')
        data = imputer.fit_transform(data)
    
    # split into training and testing, 80/20 split
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=19)

    # call bagging and random forest classifiers
    base_rt, bagging_rt, base_acc, bagging_acc = bagging(x_train, y_train, x_test, y_test)
    rf_rt, rf_acc = random_forest(x_train, y_train, x_test, y_test)

    # base classifier
    print('Base Classifier')
    print('---------------------------------')
    print(f'Training Runtime: {base_rt:.4f} seconds')
    print(f'Accuracy: {base_acc:.4f}\n')
    
    # random forest classifier
    print('Random Forest Classifier')
    print('---------------------------------')
    print(f'Training Runtime: {rf_rt:.4f} seconds')
    print(f'Accuracy: {rf_acc:.4f}\n')
    
    # bagging classifier
    print('Bagging Classifier')
    print('---------------------------------')
    print(f'Training Runtime: {bagging_rt:.4f} seconds')
    print(f'Accuracy: {bagging_acc:.4f}\n')

if __name__ == '__main__':
    main()