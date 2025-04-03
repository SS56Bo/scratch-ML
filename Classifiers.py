import numpy as np
import math


class LogisticRegression:
    def __init__(self, learn_rate, epochs):
        self.self_learning_rate = learn_rate
        self.epoch = epochs
        self.weight = None
        self.bias = None

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))