# Deric Shaffer
# CS487 - HW5
# Due Date - March 29th. 2024

import time
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# to fix the ssl verify failed error I keep running into with fetch imports
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# dataset
from sklearn.datasets import fetch_california_housing

# function definitions
def line_reg(x1, x2, y1, y2):
    lr = LinearRegression()
    #lr = LinearRegression(positive=True)

    # training
    start_time = time.time()
    lr.fit(x1, y1)
    end_time = time.time()

    # testing/predicting
    lr_pred = lr.predict(x2)

    # analysis
    lr_mse = mean_squared_error(y2, lr_pred)
    lr_r2 = r2_score(y2, lr_pred)

    print(f'Linear Regression Fitting Time: {end_time - start_time:.4f}')
    print(f'Linear Regression Mean Squared Error: {lr_mse:.4f}')
    print(f'Linear Regression R2 Score: {lr_r2:.4f}')


def ransac_reg(x1, x2, y1, y2):
    #rs = RANSACRegressor()
    rs = RANSACRegressor(residual_threshold=19)
    
    # training
    start_time = time.time()
    rs.fit(x1, y1)
    end_time = time.time()

    # testing/predicting
    rs_pred = rs.predict(x2)

    # analysis
    rs_mse = mean_squared_error(y2, rs_pred)
    rs_r2 = r2_score(y2, rs_pred)

    print(f'RANSAC Regression Fitting Time: {end_time - start_time:.4f}')
    print(f'RANSAC Regression Mean Squared Error: {rs_mse:.4f}')
    print(f'RANSAC Regression R2 Score: {rs_r2:.4f}')


def ridge_reg(x1, x2, y1, y2):
    ri = Ridge()
    #ri = Ridge(solver='lsqr')
    
    # training
    start_time = time.time()
    ri.fit(x1, y1)
    end_time = time.time()

    # testing/predicting
    ri_pred = ri.predict(x2)

    # analysis
    ri_mse = mean_squared_error(y2, ri_pred)
    ri_r2 = r2_score(y2, ri_pred)

    print(f'Ridge Regression Fitting Time: {end_time - start_time:.4f}')
    print(f'Ridge Regression Mean Squared Error: {ri_mse:.4f}')
    print(f'Ridge Regression R2 Score: {ri_r2:.4f}')


def lasso_reg(x1, x2, y1, y2):
    ls = Lasso()
    #ls = Lasso(alpha=10)
    
    # training
    start_time = time.time()
    ls.fit(x1, y1)
    end_time = time.time()

    # testing/predicting
    ls_pred = ls.predict(x2)

    # analysis
    ls_mse = mean_squared_error(y2, ls_pred)
    ls_r2 = r2_score(y2, ls_pred)

    print(f'Lasso Regression Fitting Time: {end_time - start_time:.4f}')
    print(f'Lasso Regression Mean Squared Error: {ls_mse:.4f}')
    print(f'Lasso Regression R2 Score: {ls_r2:.4f}')


def elastic_net(x1, x2, y1, y2):
    en = ElasticNet()
    #en = ElasticNet(alpha=10)

    
    # training
    start_time = time.time()
    en.fit(x1, y1)
    end_time = time.time()

    # testing/predicting
    en_pred = en.predict(x2)

    # analysis
    en_mse = mean_squared_error(y2, en_pred)
    en_r2 = r2_score(y2, en_pred)

    print(f'Elastic Net Regression Fitting Time: {end_time - start_time:.4f}')
    print(f'Elastic Net Regression Mean Squared Error: {en_mse:.4f}')
    print(f'Elastic Net Regression R2 Score: {en_r2:.4f}')



def main():
    # ask user for input
    print('\tWhich Function Do You Want To Run?')
    print('--------------------------------------------------')
    print('\ttype \'lr\' for Linear Regression')
    print('\ttype \'ran\' for RANSAC Regression')
    print('\ttype \'rid\' for Ridge Regression')
    print('\ttype \'las\' for Lasso Regression')
    print('\ttype \'enet\' for Elastic Net Regression\n')

    # end='' is added so the user input is on the same line (for formatting purposes)
    print('Selected Function: ', end='')
    c = input()
    print('\n')

    # input options
    function_options = ['lr', 'ran', 'rid', 'las', 'enet']
    
    # error check user input
    if c not in function_options:
        print('Invalid Function Selected')

    # load dataset
    cali_data = fetch_california_housing()
    x = cali_data.data
    y = cali_data.target

    # split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=19)

    # run function on training and testing sets
    if c == function_options[0]:
        # linear regression
        line_reg(x_train, x_test, y_train, y_test)

    elif c == function_options[1]:
        # ransac regression
        ransac_reg(x_train, x_test, y_train, y_test)

    elif c == function_options[2]:
        # ridge regression
        ridge_reg(x_train, x_test, y_train, y_test)

    elif c == function_options[3]:
        # lasso regression
        lasso_reg(x_train, x_test, y_train, y_test)

    elif c == function_options[4]:
        # elastic net regression
        elastic_net(x_train, x_test, y_train, y_test)

if __name__ == '__main__':
    main()