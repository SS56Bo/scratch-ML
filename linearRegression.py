import numpy as np
import matplotlib.pyplot as plot
import math

X_train = np.array([2.5, 3.6, 4.5, 7.8, 5.6, 6.2, 5.6, 5.2 ])  # our primary component
Y = np.array([100, 115, 111, 150, 142, 147, 121, 85])   #for loss function
Y_train = np.array(Y, dtype=float) # for conversion into float

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

# gradient calculation 
def ComputeGradient(X_train, Y_train, w, b):
    N=X_train.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(N):
        f_wb = w*X_train[i]+b
        dj_dw_i = (f_wb-Y_train[i])*X_train[i]
        dj_db_i = f_wb-Y_train[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw/N
    dj_db = dj_db/N

    return dj_dw, dj_db

# calculate gradient descent
def GradientDescent(X_train, Y_train, w_in, b_in, alpha, num_iterations):
    """
    performs gradient descent to find a better fit for w,b. Updates w,b 
    by taking numberof iterations with learning rate alpha
    """
    #an array to store lost function J values & also for the weights
    J_hist = []
    weights_hist = []
    x= X_train
    y= Y_train
    b=b_in
    w=w_in

    #will have to update the option to change the number of iterations
    for i in range(num_iterations):
        dj_dw, dj_db = ComputeGradient(x,y,w,b)

        b=b-alpha*dj_db
        w=w-alpha*dj_dw

        if i<num_iterations:
            J_hist.append(CostFunction(x,w,b))
            weights_hist.append([w,b])

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iterations/10) == 0:
            print(f"Iteration {i:4}: Cost {J_hist[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
            
    return w,b,J_hist,weights_hist
alpha_tmp=1.0e-2
w_final, b_final, Jhist, weightHist = GradientDescent(X_train, Y_train, 0, 0, alpha_tmp, 10000)
print(f"W_final:{w_final}, b_final:{b_final}")
    

#data visualization using Matplotlib
# y_recv = linearRegression(X_train, w, b)
# plot.plot(X_train, y_recv, c='b')
# plot.scatter(X_train, Y_train, c='g', marker="X")
# plot.title("Linear Regression")  #for the title
# plot.ylabel("Y-label")
# plot.xlabel("X-label")
# plot.show()