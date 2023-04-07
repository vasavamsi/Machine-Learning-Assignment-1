import numpy as np

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here
    coun = 0
    y = y.values
    y_hat = y_hat.values
    for i in range(np.shape(y)[0]):
        if y_hat[i] == y[i]:
            coun += 1
    acc = coun/np.shape(y)[0]
    return acc

    #pass

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """

    coun = 0
    count_y_hat = 0
    for i in range(y.size):
        if y_hat[i] == cls:
            count_y_hat += 1
        if y_hat[i] == cls and y == cls:
            coun += 1

    precise = coun/count_y_hat
    return precise

    #pass

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    coun = 0
    count_y_cls = 0
    for i in range(y.size):
        if y[i] == cls:
            count_y_cls += 1
        if y_hat[i] == cls and y[i] == cls:
            coun += 1
    output = coun/count_y_cls
    return output
    #pass

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    output = ((sum((y-y_hat)**2))/y.size)**(1/2)
    return output
    #pass

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    output = (sum(abs(y-y_hat)))/y.size
    return output

    #pass
