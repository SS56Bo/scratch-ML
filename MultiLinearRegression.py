import numpy as np
import math

def MultiLinearRegression(X_train):
    N = X_train.shape[0]
    w = np.zeros(N)
    w[0]=1
    f_wb = np.zeros(N)
    b = 0
    for i in range(N):
        f_wb[i] = np.dot(w[i],X_train[i])+b

    return f_wb

def CostFunction(X, y, w, b):
    """
        Computes the Mean Squared Error (MSE)
        Arguments:
            X: Features matrix of shape (m, n)
            y: Target vector of shape (m,)
            w: Weight vector of shape (n,)
            b: Bias term (scalar)
        Returns:
            Mean Squared Error (MSE)
    """
    N = X.shape[0]
    predict_y = np.dot(w,y) + b
    error = predict_y - y
    cost = (1/(2*N))*np.sum(error**2)
    return cost

def GradientDescentCalc(X, y, w, b):
    return "returns gradient descent calcultion"

X_TRAIN = np.array([4,8,8,0,5], dtype='float64')
Y_TRAIN = np.array([6,4,7,0,8], dtype='float64')
