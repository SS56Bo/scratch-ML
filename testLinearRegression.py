#this program is for test only

from LinearModels import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
X=X.flatten()
X=np.array(X, dtype=float)
Y=np.array(Y, dtype=float)


model = LinearRegression(0, 0)
w_final, b_final, J_cost, p_cost = model.Gradient_Descent(X, Y, 0, 0, 1.0e-2, 10000)
model = LinearRegression(w_final, b_final)
y_predict = model.Regression(X)
print(f"X set: {X}")
print(f"Y set: {Y}")
print(f"Y prediction set: {y_predict}")
plt.scatter(X, Y, marker="o")
plt.plot(X, y_predict, c="r")
plt.xlabel("Years of Experience")
plt.ylabel("Current Salary granted")
plt.title("Years vs Salary")
plt.show()