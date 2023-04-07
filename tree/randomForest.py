#from .base import DecisionTree
from sklearn import tree
import pandas as pd
import numpy as np
import random

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        #taking into account the criterion
        if criterion == 'information_gain':
            criteria = 'entropy'
        else:
            criteria = 'gini'
            
        self.dtree = tree.DecisionTreeClassifier( criterion = criteria)
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.fin_fit = range(0,self.n_estimators)
        self.feature_select_list = range(0,n_estimators)
        #pass

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        y = y.array
            
        #Starting fitting for trees with different features taken into account 
        i = 0
        while i < self.n_estimators:
            
            #Getting the features to generate the decision tree with random features
            features = np.array(X.columns)
            row,col = np.shape(X.values)
            random.seed(i)
            no_features = random.randrange(1, col, 1)
            np.random.seed(i)
            rand_features = np.random.choice(features, size = no_features, replace=False)
            rand_features.sort()
            self.feature_select_list[i] = rand_features #saving the features for using them in fitting
            data = X[rand_features]
            
            #creating the decision tree using these features
            this_fit = self.dtree.fit(data.values, y)
            self.fin_fit[i] = this_fit
            i+=1
        print(self.feature_select_list)
        #pass

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        row,col = np.shape(X.values)
        
        for sample in range(0,row):
            data = X.loc[sample]
            i = 0
            predictions_list = []
            while i < self.n_estimators:
                fin_data = data[self.feature_select_list[i]]
                print('the final data is' ,fin_data)
                print(fin_data.values.reshape(1,-1))
                prediction = self.fin_fit[i+1].predict(fin_data.values.reshape(1,-1))
                print('the prediction is',prediction)
                predictions_list.append(prediction)
                i += 1
                
            predictions_list = pd.Series(predictions_list)
            print(predictions_list.value_counts())
                
            
                
            
             
            
        #pass

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        pass



class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''

        pass

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        pass

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pass

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        pass
