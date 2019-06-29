# imports
import os
import __root__
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

# data preprocessing
# importing the dataset
resource = os.path.join(__root__.PROJECT_ROOT, 'data/regression/multiple_linear_regression/')
dataset = pd.read_csv(resource + '50_Startups.csv')
# spliting the dataset into independent variable and depended variable
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# categorize the cities column into numerical data
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])], remainder='passthrough')
x = ct.fit_transform(x)

# avoiding the dummy variable trap, the sklearn is going to take care of that but...
x = x[:, 1:]

# spliting the dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# feature scaling: there is not need to feature scaling because the library is going to take care of that
# Fitting multiple linear regression to the training set
regressor = LinearRegression()
regressor.fit(X=x_train, y=y_train)

# predicting the test set results
y_pred = regressor.predict(x_test)


# building the optimal model using backward elimination
# step 1
x = np.append(arr=np.ones((50, 1)).astype(int), values=x, axis=1)
# step 2
x_opt = x[:, [0, 1, 2, 3, 4, 5]].astype('float64')
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
# step 3
print(regressor_OLS.summary())
# in summary it is observed that independet variable x2 is having highest p value. we must remove it

# step 2
x_opt = x[:, [0, 1, 3, 4, 5]].astype('float64')
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
# step 3
print(regressor_OLS.summary())


# in summary it is observed that independet variable x1 is having highest p value. we must remove it
# step 2
x_opt = x[:, [0, 3, 4, 5]].astype('float64')
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
# step 3
print(regressor_OLS.summary())


# in summary it is observed that independet variable x2 is having highest p value. we must remove it
# step 2
x_opt = x[:, [0, 3, 5]].astype('float64')
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
# step 3
print(regressor_OLS.summary())


# in summary it is observed that independet variable x5 is having highest p value. we must remove it
# step 2
x_opt = x[:, [0, 3]].astype('float64')
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
# step 3
print(regressor_OLS.summary())
