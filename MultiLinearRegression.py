import numpy as np
import math

def MultilinearRegression(X_train):
    N = X_train.shape[0]
    w = np.zeros(N)
    f_wb = np.zeros(N)
    b = 0
    for i in range(N):
        f_wb[i] = np.dot(w[i],X_train[i])+b

    return f_wb

X_TRAIN = np.array([4,8,8,0,5])
print(MultilinearRegression(X_TRAIN))
