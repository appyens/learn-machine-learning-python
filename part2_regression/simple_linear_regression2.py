import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# %matplotlib inline
from sklearn.datasets import load_boston

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
    pass