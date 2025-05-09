import cupy as cp
import math

class LinearModel:
    class LinearRegression:

        def __init__(self):
            self.w=0.0
            self.b=0.0

        def fit_transform(self, X):
            X = cp.asarray(X, dtype=cp.float32)
            return self.w * X + self.b
        
        def predict(self, X):
            X = cp.asarray(X, dtype=cp.float32)
            return self.w * X + self.b
        
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
        
        def GradientOptimize(self, X, y, alpha, epochs):
            X_train = cp.asarray(X, dtype=cp.float32)
            y_train = cp.asarray(y, dtype=cp.float32)
            J_hist = []
            b_hist = []

            for i in range(epochs):
                dj_dw, dj_db = self.__ComputeGradient(X_train, y_train)

                self.w -= alpha*dj_dw
                self.b -= alpha*dj_db

                if i<epochs:
                    J_hist.append(self.__CostFunction(X_train, y_train))
                    b_hist.append([self.w, self.b])

            
                if i % max(1, math.ceil(epochs / 1000)) == 0:
                    print(f"Iterations {i: 5}: Cost: {J_hist[-1]:0.2e} ",
                          f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e} ",
                          f"w: {self.w: 0.3e}, b: {self.b: 0.5e} "
                    )
                
            return self.w, self.b
            
    
    class RidgeRegression:

        def __init__(self):
            self.w=0.0,
            self.b=0.0

        def fit_transform(self, X):
            X = cp.asarray(X, dtype=cp.float32)
            return self.w*X+self.b
        
        def predict(self, X):
            X = cp.asarray(X, dtype=cp.float32)
            return self.w*X+self.b
        
        def __CostFunction(self, X, y, alpha):
            #basically will use Ridge adjustment
            X = cp.asarray(X, dtype=cp.float32)
            y_in = cp.asarray(y_in, dtype=cp.float32)
            N = X.shape[0]
            err = self.fit_transform(X)-y_in
            residuals = cp.sum(cp.power(err, 2))
            ridge_penalty = (alpha*cp.power(self.w,2))
            total_cost = (residuals+ridge_penalty)/N
            return total_cost
        
        def _ComputeGradientDescent(self, X, y_in, alpha):
            X = cp.asarray(X, dtype=float)
            y_in = cp.asarray(y_in, dtype=float)
            N = X.shape[0]
            Y_predict = self.fit_transform(X)
            err = Y_predict-y_in
            dj_dw = (2/N)*cp.sum((err)*X)+(2*alpha*self.w)  # L2 Regression
            dj_db = (2/N)*cp.sum((err))
            return dj_dw, dj_db
        
        def GradientOptimization(self, X, y, w_in=0.0, b_in=0.0, alpha=1, eta=0.01, epoch=1000):
            self.w = w_in
            self.b = b_in
            X_train = X
            y_train = y
            J_hist = []
            weight_hist = []