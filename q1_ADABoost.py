"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *
import pandas as pd
from ensemble.ADABoost import AdaBoostClassifier
#from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
from sklearn import tree
from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")
criteria = 'information_gain'
#tree = DecisionTree(criterion=criteria)
stump = tree.DecisionTreeClassifier(max_depth = 1, criterion = 'entropy')
Classifier_AB = AdaBoostClassifier(base_estimator=stump, n_estimators=n_estimators )
Classifier_AB.fit(X, y) 
y_hat = Classifier_AB.predict(X)

#[fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))


##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features
X = pd.read_csv('iris.data')
features = ['sepal_length', 'sepal_width']
y = X['species']
y = y.values
X = X[features]
for i in range(0,len(y)):
    if y[i] != 'Iris-virginica':
        y[i] = 'Iris-nonvirginica'
X = X.values
X = pd.DataFrame(X)
y = pd.Series(y)
stump = tree.DecisionTreeClassifier(max_depth = 1, criterion = 'entropy')
Classifier_AB = AdaBoostClassifier(base_estimator=stump, n_estimators=n_estimators )
Classifier_AB.fit(X, y) 
y_hat = Classifier_AB.predict(X)

print(y_hat)