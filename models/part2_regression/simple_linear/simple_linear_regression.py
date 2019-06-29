import os
import __root__
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

resource = os.path.join(__root__.PROJECT_ROOT, 'data/regression/simple_linear_regression/')
csv = os.path.join(resource, 'Salary_Data.csv')
# reading dataset with read.csv
dataset = pd.read_csv(csv)

# independent variable, predictor, attribute
x = dataset.iloc[:, :-1].values
# dependent variable, response, label
y = dataset.iloc[:, 1].values

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# fitting simple linear regression to the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# to retrieve the intercept
print(regressor.intercept_)
# to retrieve the slope
# This means that for every one unit of change in X , the change in the Y is about 9312%
print(regressor.coef_)

# predicting the test set results
# Now that we have trained our algorithm, it’s time to make some predictions.
# To do so, we will use our test data and see how accurately our algorithm predicts the percentage score.
y_pred = regressor.predict(x_test)

# Now compare the actual output values for X_test with the predicted values, execute the following script:
compare = pd.DataFrame({"Actual": y_test, 'Predicted': y_pred})

# predicting train set results
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

# The final step is to evaluate the performance of the algorithm. This step is particularly important
# to compare how well different algorithms perform on a particular dataset.
# For regression algorithms, three evaluation metrics are commonly used:

# Mean Absolute Error (MAE) is the mean of the absolute value of the errors. It is calculated as:
# Mean Squared Error (MSE) is the mean of the squared errors and is calculated as:
# Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors:

print("Mean absolute error: ", metrics.mean_absolute_error(y_test, y_pred))
print("mean squared error: ", metrics.mean_squared_error(y_test, y_pred))
print("Root mean square error: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))