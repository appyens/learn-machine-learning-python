import os
import __root__
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

resource = os.path.join(__root__.PROJECT_ROOT, 'data/simple_linear_regression/')
csv = os.path.join(resource, 'Salary_Data.csv')
# reading dataset with read.csv
dataset = pd.read_csv(csv)
x = dataset.iloc[:, :-1].values
# dependent variable
y = dataset.iloc[:, 1].values
# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# fitting simple linear regression to the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)
# predicting the test set results
y_pred = regressor.predict(x_test)
x_pred = regressor.predict(x_train)

# visualising the training set results
plt.scatter(x_train, y_train, color='red')
# plotting regression line (predicted values)
plt.plot(x_train, x_pred, color='blue')
plt.title("Salary Vs Experience (Training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

# visualising the test set results
plt.scatter(x_test, y_test, color='red')
# plotting regression line
plt.plot(x_train, x_pred, color='blue')
plt.title("Salary Vs Experience (Test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()
