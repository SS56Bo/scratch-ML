import numpy as np

class StandardScaler():
    def __init__(self):
        self.mean = None
        self.std = None
    
    def calcFit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
    
    def Ztransform(self, X):
        return (X-self.mean)/self.std
    
    def fit_transform(self, X):
        self.calcFit(X)
        return self.Ztransform(X)