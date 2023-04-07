import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, fit_intercept=True, method='normal'):
        '''

        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        :param method:
        '''
        self.fit_intercept = fit_intercept
        self.param = []
        self.y_predict = []
        self.y = []
        pass

    def fit(self, X, y):
        '''
        Function to train and construct the LinearRegression
        :param X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        :param y: pd.Series with rows corresponding to output variable (shape of Y is N)
        :return:
        '''
        X = X.values
        y = y.values
        self.y = y
        row,col = np.shape(X) # taking into account the shape of X
        
        #considering the intercept option
        if self.fit_intercept == True:
            intercept = np.ones((row,col+1))
            intercept[:, 1:(col+1)] = X
            fin_X = intercept
        else:
            fin_X = X
        
        #Applying the normal equation
        X_trans = np.transpose(fin_X)
        product_1 = X_trans.dot(fin_X)
        product_1_inv = np.linalg.inv(product_1)
        product_2 = product_1_inv.dot(X_trans)
        self.param = product_2.dot(y)
        
        #pass

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point
        :param X: pd.DataFrame with rows as samples and columns as features
        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        X = X.values
        row,col = np.shape(X) # taking into account the shape of X
        #print(self.param)
        #considering the intercept option
        if self.fit_intercept == True:
            intercept = np.ones((row,col+1))
            intercept[:, 1:(col+1)] = X
            fin_X = intercept
        else:
            fin_X = X
        
        #getting the prediction
        self.y_predict = fin_X.dot(self.param)
        return pd.Series(self.y_predict)
        #pass

    def plot_residuals(self):
        """
        Function to plot the residuals for LinearRegression on the train set and the fit. This method can only be called when `fit` has been earlier invoked.

        This should plot a figure with 1 row and 3 columns
        Column 1 is a scatter plot of ground truth(y) and estimate(\hat{y})
        Column 2 is a histogram/KDE plot of the residuals and the title is the mean and the variance
        Column 3 plots a bar plot on a log scale showing the coefficients of the different features and the intercept term (\theta_i)
        """
        if self.fit_intercept == True:
            x = range(self.y.shape[0])
            #plt.figure(1)
            f1, (ax1,ax2) = plt.subplot(1,2)
            ax1.scatter(x,self.y)
            ax1.scatter(x,self.y_predict)
            x = range(self.param.shape[0])
            #plt.figure(2)
            ax2.bar(x,self.param, log=True)
        else:
            x = range(self.y.shape[0])
            #plt.figure(3)
            f2, (ax_1,ax_2) = plt.subplot(1,2)
            ax_1.scatter(x,self.y)
            ax_1.scatter(x,self.y_predict)
            x = range(self.param.shape[0])
            #plt.figure(4)
            ax_2.bar(x,self.param, log=True)
    def plot_param(self):
        x = range(self.param.shape[0])
        plt.figure(0)
        plt.scatter(x, np.transpose(self.param))
        
        #pass
