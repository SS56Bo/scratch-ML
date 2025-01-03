import numpy as np
import math

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

    def __computeGradient(self, X, y_test):
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
    
    def Gradient_Descent(self, X_train, y_train, w_in, b_in, alpha, num_iters):
        """
        performs gradient descent to find a better fit for w,b. Updates w,b 
        by taking numberof iterations with learning rate alpha
        """
        self.w=w_in
        self.b=b_in
        J_hist=[]
        p_hist=[]
        x=X_train
        y=y_train

        for i in range(num_iters):
            dj_dw, dj_db = self.__computeGradient(x, y)

            #updating the weights in the algorithm
            self.b-=alpha*dj_db
            self.w-=alpha*dj_dw
            
            if i<num_iters:
                J_hist.append(self.CostFunctionCalc(x, y))
                p_hist.append([self.w, self.b])
            
            #for visualization purpose only
            if i%math.ceil(num_iters/10)==0:
                print(f"Iterations {i: 5}: Cost: {J_hist[-1]:0.2e} ",
                      f"dj_dw: {dj_dw: 0.3e}, dj_sdb: {dj_db: 0.3e} ",
                      f"w: {self.w: 0.3e}, b: {self.b: 0.5e} "
                      )
        
        return self.w, self.b, J_hist, p_hist

class PolynomialRegression:
    def __init__(self, w, b):
        self.w = float(w)
        self.b = float(b)