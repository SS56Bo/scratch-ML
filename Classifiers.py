import numpy as np
import math


class LogisticRegression:
    def __init__(self, learn_rate, epochs):
        self.learning_rate = learn_rate
        self.epoch = epochs
        self.weight = None
        self.bias = None

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def CostFunction(self,y, predict):
        num_samples = len(y)
        cost = -(1/num_samples)*np.sum(y*np.log(predict)+(1-y)*np.log(1-predict))
        return cost
    
    def GradientDescentCalc(self, X, y, predict):
        num_samples = X.shape[0]
        dw = (1/num_samples)*np.dot(X.T, (predict-y))
        dj = (1/num_samples)*np.sum(predict-y)
        self.weight -= self.learning_rate*dw
        self.bias -= self.learning_rate*dj