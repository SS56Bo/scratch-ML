import cupy as cp
import math

class ml_model:
    class LinearModel:
        class LinearRegression:

            def __init__(self):
                self.w=0.0,
                self.b=0.0

            def fit_transform(self, X):
                X = cp.asarray(X, dtype=cp.float32)
                return self.w*X+self.b
        
            def __CostFunction(self, X, y):
                X = cp.asarray(X, dtype=cp.float32)
                y = cp.asarray(y, dtype=cp.float32)
                N = X.shape[0]
                y_error = self.fit_transform(X)-y
                costSum = cp.sum(cp.power(y_error, 2))/N
                return costSum
        
            def __ComputeGradient(self, X, y):
                X = cp.asarray(X, dtype=cp.float32)
                y = cp.asarray(y, dtype=cp.float32)
                N = X.shape[0]
                err = self.fit_transform(X)-y
                dj_dw_i = cp.sum(err*X)/N
                dj_db_i = cp.sum(err)/N
                return dj_dw_i, dj_db_i
        
            def Gradientoptimize(self, X, y, alpha, epochs):
                X_train = cp.asarray(X, dtype=cp.float32)
                y_train = cp.asarray(y, dtype=cp.float32)
                N = X.shape[0]
                J_hist = []
                b_hist = []

                for i in range(epochs):
                    dj_dw, dj_db = self.__ComputeGradient(X, y)

                    self.w -= alpha*dj_dw
                    self.b -= alpha*dj_db

                    if i<epochs:
                        J_hist.append(self.__CostFunction(X_train, y_train))
                        b_hist.append([self.w, self.b])

                    #for visualization purpose only
                    if i%math.ceil(epochs/50)==0:
                        print(f"Iterations {i: 5}: Cost: {J_hist[-1]:0.2e} ",
                      f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e} ",
                      f"w: {self.w: 0.3e}, b: {self.b: 0.5e} "
                        )
                    return self.w, self.b
            
    
