import numpy as np

class LinearRegression:
    
    def __init__(self, w, b):
        self.w = float(w)
        self.b = float(b)

    def Regression(self, X):
        X= np.array(X, dtype=float)
        N = X.shape[0]
        f_wb = np.zeros(N)
        for i in range(N):
            f_wb[i]=np.dot(self.w, X[i])+self.b
        
        return f_wb

    def CostFunctionCalc(self, X, test):
        X=np.array(X, dtype=float)
        test=np.array(test, dtype=float)
        costSum = 0
        N = X.shape[0]
        f_wb = np.zeros(N)
        for i in range(N):
            f_wb[i]=np.dot(self.w, X[i])+self.b
            cost = (1/N)*((test[i]-f_wb[i])**2)
            costSum += cost
        
        return costSum
