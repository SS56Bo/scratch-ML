import numpy as np
import matplotlib.pyplot as plot

X_train = np.array([2.5, 3.6, 4.5, 7.8, 5.6, 6.2, 5.6 ])  # our primary component
Y = np.array([100, 115, 111, 150, 142, 147, 121])   #for loss function
Y_train = np.dot(Y, 1.0) # for conversion into float

#training set
print(f"X Trained Data: {X_train}")
print(f"Y Trained Data: {Y_train}\n")

#Training set shape
print(f"Shape of training set: {X_train.shape}")
print(f"Number of training set examples: {X_train.shape[0]}\n")

#dataset
for i in range(X_train.shape[0]):
    print (f"[X^{i}, Y^{i}] = [{X_train[i]}, {Y_train[i]}]")

#weights for linear regression
w = 0
b = 0

#data visualization using Matplotlib
plot.scatter(X_train, Y_train, c='g', marker="2")
plot.title("X vs. Y")  #for the title
plot.ylabel("Y-label")
plot.xlabel("X-label")
plot.show()

# equation for optimization

# loss function 

# gradient descent 