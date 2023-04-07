import numpy as np
import math
import pandas as pd


class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators=3): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.n_estimators = n_estimators
        self.labels = []
        self.fin_dec_stumps = []
        self.fin_dec_labels = []
        self.fin_err_array = []
        self.fin_Alpha_array = []
        #pass

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        no_samples = len(X.index)
        self.labels = y.unique() #Getting the array of labels

        X['labels'] = y
        X['sample_weights'] = np.ones((no_samples,1))
        
        features = X.columns
        
        for est in range(0,self.n_estimators):
            pre_info_gain = 0
            for feature in range(0,len(features)-2):
                df = X.sort_values(features[feature]) #Sorting the features accordingly
                data = df.values
                #Calculating the overall entropy
                sum_0 = 0
                sum_1 = 0
                for i in range(0, np.shape(data)[0]):
                    if data[i,-2] == self.labels[0]:
                        sum_0 += data[i,-1]
                    else:
                        sum_1 += data[i,-1]
                prob_0 = sum_0/(sum_0+sum_1)
                prob_1 = sum_1/(sum_0+sum_1)
                
                if prob_0 == 0:
                    S = -(prob_1*math.log(prob_1,2))
                elif prob_1 == 0:
                    S = -(prob_0*math.log(prob_0,2))
                else:
                    S = -(prob_0*math.log(prob_0,2) + prob_1*math.log(prob_1,2))
                        
                # for getting the best splits
                pre_ent = 999
                for i in range(1, np.shape(data)[0]-1):
                    data_above = data[0:i, :]
                    data_below = data[i+1:, :]
                    
                    #Calculating overall entropy
                    #Calculating the entropy for the data_above
                    sum_0 = 0
                    sum_1 = 0
                    for k in range(0, np.shape(data_above)[0]):            
                        if data_above[k,-2] == self.labels[0]:
                            sum_0 += data_above[k,-1]
                        else:
                            sum_1 += data_above[k,-1]
                    prob_0 = sum_0/(sum_0 + sum_1)
                    prob_1 = sum_1/(sum_0 + sum_1)
                    
                    if prob_0 == 0:
                        ent_above = -(prob_1*math.log(prob_1,2))
                    elif prob_1 == 0:
                        ent_above = -(prob_0*math.log(prob_0,2))
                    else:
                        ent_above = -(prob_0*math.log(prob_0,2) + prob_1*math.log(prob_1,2))
                    
                    #getting the best label
                    if prob_0 > prob_1:
                        label_above = self.labels[0]
                    else:
                        label_above = self.labels[1]
                    
                    #Calculating the entropy for the data_below
                    sum_0 = 0
                    sum_1 = 0
                    for k in range(0, np.shape(data_below)[0]):            
                        if data_below[k,-2] == self.labels[0]:
                            sum_0 += data_below[k,-1]
                        else:
                            sum_1 += data_below[k,-1]
                    prob_0 = sum_0/(sum_0 +sum_1)
                    prob_1 = sum_1/(sum_0 +sum_1)
                    
                    if prob_0 == 0:
                        ent_below = -(prob_1*math.log(prob_1,2))
                    elif prob_1 == 0:
                        ent_below = -(prob_0*math.log(prob_0,2))
                    else:
                        ent_below = -(prob_0*math.log(prob_0,2) + prob_1*math.log(prob_1,2))
                    
                    overall_ent = (np.shape(data_above)[0])*ent_above + (np.shape(data_below)[0])*ent_below        
                    overall_ent = overall_ent/(np.shape(data)[0])
                    
                    #getting the best label
                    if prob_0 > prob_1:
                        label_below = self.labels[0]
                    else:
                        label_below = self.labels[1]
                        
                    ##Getting the best split
                    if overall_ent <= pre_ent:
                        best_split = (data[i, features[feature]] + data[i+1, features[feature]])/2
                        parted_at = i
                        pre_ent = overall_ent
                        best_above_ent = ent_above
                        best_below_ent = ent_below
                
                ##Selecting the best feature from the information gain
                info_gain = S - (((parted_at+1)*best_above_ent + (np.shape(data)[0]-parted_at-1)*best_below_ent)/(np.shape(data)[0]))
                
                if info_gain > pre_info_gain:
                    best_feature = feature
                    dec_stump = [feature, best_split]
                    dec_labels = [label_above, label_below]  #first element is if 'val < best_split' and next for 'val >= best_split'
                    pre_info_gain = info_gain
                
            self.fin_dec_stumps.append(dec_stump)
            self.fin_dec_labels.append(dec_labels)
            
            ##Getting the prediction values
            y_pred = []
            for sample in range(0, np.shape(X.values)[0]):
                if X.values[sample, dec_stump[0]] < dec_stump[1]:
                    y_pred.append(dec_labels[0])
                else:
                    y_pred.append(dec_labels[1])
            y_pred = np.array(y_pred)
            
            ##Error calculation
            wrong_pred_sum = 0
            tot_pred_sum = 0
            for out in range(0, len(y)):
                tot_pred_sum += data[out,-1]
                if y_pred[out] == y.array[out]:
                    wrong_pred_sum += data[out,-1]
            error = wrong_pred_sum / tot_pred_sum
            self.fin_err_array.append(error)
                
            ##Alpha calculatiion
            Alpha = 0.5*(math.log((1-error)/error))
            self.fin_Alpha_array.append(Alpha)
            
            ##Updating the weights
            for out in range(0, len(y)):
                if y_pred[out] == y.array[out]:
                    data[out,-1] = data[out,-1]/math.exp(Alpha)
                else:
                    data[out,-1] = data[out,-1]*math.exp(Alpha)
            X = pd.DataFrame(data)

        #pass

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        data = X.values
        y_hat = []
        for sample in range(0,np.shape(data)[0]):
            fin_pred = 0
            for est in range(0, self.n_estimators):
                
                #Making the prediction for considered estimator
                decision = self.fin_dec_stumps[est]
                result = self.fin_dec_labels[est]
                if data[sample,decision[0]] < decision[1]:
                    if result[0] == self.labels[0]:
                        pred=1
                    else:
                        pred=-1
                else:
                    if result[1] == self.labels[0]:
                        pred=1
                    else:
                        pred=-1
                
                #Calculating the approx prediction
                fin_pred += self.fin_Alpha_array[est]*pred
            
            #Getting the final prediction
            if fin_pred > 0:
                y_hat.append(self.labels[1])
            else:
                y_hat.append(self.labels[0])
            
        return pd.Series(y_hat, dtype = 'category')
        #pass

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        
        pass
