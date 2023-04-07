import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.fin_fits = []
        self.labels = []
        #pass

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        data_indices = X.index
        self.labels = y.unique()
        i = 0
        while i < self.n_estimators:
            
            #Randomizing the data
            np.random.seed(i)
            sampled_indices = np.random.choice(data_indices, size=len(data_indices), replace=True)
            sampled_data = X.loc[sampled_indices]
            sampled_target = y.loc[sampled_indices]
            
            #Plotting the values for question 3(b).....Comment it out when not required
            print('classification plot for sampled data set for iteration', i)
            plt.figure(i)
            plt.scatter(sampled_data.values[:,0], sampled_data.values[:,1], sampled_target.array==1, c='y')
            plt.scatter(sampled_data.values[:,0], sampled_data.values[:,1], sampled_target.array==0, c='b')
            plt.show(i)
            
            #Taking different deicison boundaries into account
            fits = self.base_estimator.fit(sampled_data.values,sampled_target.array)
            self.fin_fits.append(fits)
            i+=1
            
        #pass

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_predict = []
        X = X.values
        row,col = np.shape(X)
        cnt_0 = 0
        cnt_1 = 0
        
        for sample in range(0,row):
            X_data = X[sample,:]
            i = 0
            prediction_array = []
            while i <= self.n_estimators:
                prediction = self.fin_fits[i].predict(X_data.reshape(1,-1))
                prediction_array.append(prediction)
                if prediction == self.labels[0]:
                    cnt_0 += 1
                else:
                    cnt_1 += 1
                i += 1
            if cnt_0 > cnt_1:
                y_predict.append(self.labels[0])
            else:
                y_predict.append(self.labels[1])

        return pd.Series(y_predict, dtype = 'category')           
        #pass

    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        pass
