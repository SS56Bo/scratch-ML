import numpy as np

class LinearRegression:
    
    def __init__(self, w, b):
        self.w = np.array(w, dtype=float)
        self.b = float(b)

    def Regression(self, X):
        X= np.array(X, dtype=float)
        N = X.shape[0]
        f_wb = np.zeros(N)
        for i in range(N):
            f_wb[i]=np.dot(self.w, X[i])+self.b
        
        return f_wb