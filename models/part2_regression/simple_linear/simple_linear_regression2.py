import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# %matplotlib inline
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

boston_dataset = load_boston()
print(boston_dataset.keys())
print(boston_dataset.DESCR)
print(boston_dataset.data)
# convert boston dataset to pandas dataframe
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()
# add MEDV feature to boston data
boston['MEDV'] = boston_dataset.target
# preprocessing taking care of missing value
# check count of missing values
boston.isnull().sum()
"""
Exploratory Data Analysis
Exploratory Data Analysis is a very important step before training the model. 
In this section, we will use some visualizations to understand the relationship of the target variable 
with other features. Let’s first plot the distribution of the target variable MEDV. 
We will use the distplot function from the seaborn library.
"""

sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()

"""
We see that the values of MEDV are distributed normally with few outliers.
Next, we create a correlation matrix that measures the linear relationships between the variables. 
The correlation matrix can be formed by using the corr function from the pandas dataframe library. 
We will use the heatmap function from the seaborn library to plot the correlation matrix.
"""
correlation_matrix = boston.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

"""
The correlation coefficient ranges from -1 to 1. If the value is close to 1, 
it means that there is a strong positive correlation between the two variables. 
When it is close to -1, the variables have a strong negative correlation.

Observations:
To fit a linear regression model, we select those features which have a high correlation with our target variable MEDV. 
By looking at the correlation matrix we can see that RM has a strong positive correlation with MEDV (0.7) 
where as LSTAT has a high negative correlation with MEDV(-0.74). An important point in selecting features for a linear 
regression model is to check for multi-co-linearity. The features RAD, TAX have a correlation of 0.91. 
These feature pairs are strongly correlated to each other. We should not select both these features together for 
training the model. Check this for an explanation. Same goes for the features DIS and AGE which have a correlation of -0.75.
Based on the above observations we will RM and LSTAT as our features. Using a scatter plot let’s see how these features vary with MEDV.
"""

plt.figure(figsize=(20, 5))
features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features), i + 1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
    plt.show()

"""
Observations:
The prices increase as the value of RM increases linearly. There are few outliers and the data seems to be capped at 50.
The prices tend to decrease with an increase in LSTAT. Though it doesn’t look to be following exactly a linear line.

Preparing the data for training the model
We concatenate the LSTAT and RM columns using np.c_ provided by the numpy library.
"""

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT','RM'])
Y = boston['MEDV']

"""
Splitting the data into training and testing sets
Next, we split the data into training and testing sets. We train the model with 80% of the samples and test with 
the remaining 20%. We do this to assess the model’s performance on unseen data. To split the data we use 
train_test_split function provided by scikit-learn library. 
We finally print the sizes of our training and test set to verify if the splitting has occurred properly.
"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

"""
Training and testing the model
We use scikit-learn’s LinearRegression to train our model on both the training and test sets.
"""

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


"""
Model evaluation
We will evaluate our model using RMSE and R2-score.
"""

# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))