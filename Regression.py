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

    def countGradient(self, X, y_test):
        N = X.shape[0]
        dj_dw = 0
        dj_db = 0
        for i in range(N):
            f_wb = self.w*X[i]+self.b
            dj_dw_i=(f_wb-y_test[i])*X[i]
            dj_db_i=f_wb-y_test[i]
            dj_dw += dj_dw_i
            dj_db += dj_db_i
        dj_db=dj_db/N
        dj_dw=dj_dw/N
        
        return dj_dw, dj_db