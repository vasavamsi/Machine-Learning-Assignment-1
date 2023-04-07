"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.bagging import BaggingClassifier
#from tree.base import DecisionTree
# Or use sklearn decision tree
from sklearn import tree
from linearRegression.linearRegression import LinearRegression

########### BaggingClassifier ###################
"""
np.random.seed(10)
N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
dtree = tree.DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
#tree = DecisionTree(criterion=criteria)
Classifier_B = BaggingClassifier(base_estimator=dtree, n_estimators=n_estimators )
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)
print(y)
print(y_hat)
#[fig1, fig2] = Classifier_B.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))
"""
#############Question 3(b)#############################

data = pd.read_excel('Q3_b_dataset.xlsx', header=1) #Dataset has been created manually in excel sheet and loaded
y = pd.Series(data['label'], dtype = 'category')
X = data.drop(['label'], axis=1)
"""
X = X.values
plt.scatter(X[:,0], X[:,1], y==1, c='y')
plt.scatter(X[:,0], X[:,1], y==0, c='b')
plt.show()
"""
dtree = tree.DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
Classifier_B = BaggingClassifier(base_estimator=dtree, n_estimators=5 )
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)
print(y_hat)

#plotting the predicted classification plot

plt.figure()
plt.scatter(X.values[:,0], X.values[:,1], y_hat.values==1, c = 'y')
plt.scatter(X.values[:,0], X.values[:,1], y_hat.values==0, c = 'b')
plt.show()
