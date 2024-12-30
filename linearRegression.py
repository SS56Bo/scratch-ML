import numpy as np
import matplotlib.pyplot as plot

X_train = np.array([2.5, 3.6, 4.5, 7.8, 5.6, 6.2, 5.6 ])  # our primary component
Y = np.array([100, 115, 111, 150, 142, 147, 121])   #for loss function
Y_train = np.dot(Y, 1.0) # for conversion into float

#training set
print(f"X Training Data: {X_train}")
print(f"Y Training Data: {Y_train}\n")

#Training set shape
print(f"Shape of training set: {X_train.shape}")
print(f"Number of training set examples: {X_train.shape[0]}\n")

#dataset
for i in range(X_train.shape[0]):
    print (f"[X^({i}), Y^({i})] = [{X_train[i]}, {Y_train[i]}]")

#weights for linear regression
w = 100
b = 100


# equation for optimization
def linearRegression(X_train, w, b):
    N = X_train.shape[0]
    f_wb = np.zeros(N)
    for i in range(N):
        f_wb[i] = np.dot(w, X_train[i])+b

    return f_wb

print(linearRegression(X_train, w, b))

# loss function 
def CostFunction(X_train, w, b):
    N=X_train.shape[0]
    y_rec = np.zeros(N)
    costSum=0
    for i in range(N):
        y_rec[i]=np.dot(w,X_train[i])+b
        cost = (1/N)*((Y_train[i]-y_rec[i])**2)
        costSum+=cost

    return costSum

print(f"Cost Function: {CostFunction(X_train,w,b)}")

# gradient descent 

#data visualization using Matplotlib
y_recv = linearRegression(X_train, w, b)
plot.plot(X_train, y_recv, c='b')
plot.scatter(X_train, Y_train, c='g', marker="X")
plot.title("Linear Regression")  #for the title
plot.ylabel("Y-label")
plot.xlabel("X-label")
plot.show()