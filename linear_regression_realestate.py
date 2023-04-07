import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

data = pd.read_excel('Real_estate_valuation_data_set.xlsx')
X = data.drop(['No'], axis=1)
row,col = np.shape(X) 

start = 0
end = 81
k = 1
while k <= 5:
    print('Initiating fold for k= ', k)
    #Generating the test set
    X_test = X.values[start:end, 0:col-1]
    y_test = X.values[start:end, -1]
    
    #Generating the train set
    data_train = np.delete(X.values, range(start,end+1),0)
    X_train = data_train[:,0:col-1]
    y_train = data_train[:,-1]
    
    #Generating the parameters
    LR = LinearRegression(fit_intercept=True)
    LR.fit(pd.DataFrame(X_train), pd.Series(y_train))
    y_hat = LR.predict(pd.DataFrame(X_test))
    print(y_hat)
    LR.plot_param()
    
    #For Test MAE
    print('The Test MAE is', mae(y_hat,y_test))
    
    #For Train MAE
    y_hat = LR.predict(pd.DataFrame(X_train))
    print(y_hat)
    print('The Train MAE is', mae(y_hat, y_train))
    
    k += 1
    start = end+1
    end = end+81
    
    