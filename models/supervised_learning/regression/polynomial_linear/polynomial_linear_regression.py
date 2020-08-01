# imports
import os
from __root__ import PROJECT_ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# importing dataset
csv = os.path.join(PROJECT_ROOT, 'data/supervised_learning/regression/polynomial_linear_regression/Position_Salaries.csv')
dataset = pd.read_csv(csv)

# splitting dataset in separate variables
# make sure you make C as a matrix
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# here there is no need to splitting the dataset into training set and test set
# dataset is too small so it will not affect much

# here there is no need to feature scaling

# fitting linear regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X=X, y=Y)

# fitting polynomial regression to the dataset
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X=X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X=X_poly, y=Y)

# visualising the linear regression results
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualising the polynomial regression results
x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
