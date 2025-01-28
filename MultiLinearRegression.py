import numpy as np
import math
import matplotlib.pyplot as plt

def MultiLinearRegression(X_train):
    N = X_train.shape[0]
    w = np.zeros(N)
    w[0]=1
    f_wb = np.zeros(N)
    b = 0
    for i in range(N):
        f_wb[i] = np.dot(w[i],X_train[i])+b

    return f_wb

def CostFunction(X, y, w, b):
    """
        Computes the Mean Squared Error (MSE)
        Arguments:
            X: Features matrix of shape (m, n)
            y: Target vector of shape (m,)
            w: Weight vector of shape (n,)
            b: Bias term (scalar)
        Returns:
            Mean Squared Error (MSE)
    """
    N = X.shape[0]
    predict_y = np.dot(X,w) + b
    error = predict_y - y
    cost = (1/(2*N))*np.sum(error**2)
    return cost

def GradientDescentCalc(X, y, w, b):
    """
    Calculates the gradients of the cost function with respect to weights and bias.
        Args:
            X: Features matrix of shape (m, n)
            y: Target vector of shape (m,)
            w: Weight vector of shape (n,)
            b: Bias term (scalar)
        Returns:
            dj_dw: Gradient with respect to weights (shape: (n,))
            dj_db: Gradient with respect to bias (scalar)
    """
    N = X.shape[0]
    predict = np.dot(X, w)+b
    error = predict-y
    X=X.T
    dj_dw = (1/N)*np.dot(X, error)
    dj_db = (1/N)*np.sum(error)
    return dj_dw, dj_db

def GradientDescentOptimization(X,y,w,b, alpha, iters):
    """
        Optimizes weights and bias using gradient descent.
        Args:
            X: Features matrix of shape (m, n)
            y: Target vector of shape (m,)
            w: Initial weight vector of shape (n,)
            b: Initial bias term (scalar)
            alpha: Learning rate
            num_iters: Number of iterations
        Returns:
            w: Optimized weights
            b: Optimized bias
            cost_history: List of costs over iterations
    """
    cost_hist = []
    for i in range(iters):
        dj_dw, dj_db = GradientDescentCalc(X,y,w,b)

        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        cost = CostFunction(X,y,w,b)
        cost_hist.append(cost)

        if (i%10==0):
            print(f"Iterations {i}: {cost:.4f}")

    return w,b, cost_hist

X_TRAIN = np.array([[4, 8], [8, 5], [5, 2]], dtype='float64')  # (m, n)
Y_TRAIN = np.array([6, 4, 7], dtype='float64')  # (m,)
w_init = np.zeros(X_TRAIN.shape[1])  # Initialize weights to zeros
b_init = 0  # Initialize bias to zero
alpha = 0.01  # Learning rate
iterations = 50  # Number of iterations

# Run Gradient Descent Optimization
w_opt, b_opt, cost_hist = GradientDescentOptimization(X_TRAIN, Y_TRAIN, w_init, b_init, alpha, iterations)

print("\nOptimized Weights:", w_opt)
print("Optimized Bias:", b_opt)
