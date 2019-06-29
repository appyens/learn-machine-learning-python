from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

"""
In the following example, we will use multiple linear regression to predict the stock index price 
(i.e., the dependent variable) of a fictitious economy by using 2 independent/input variables:
"""
Stock_Market = {'Year': [2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
                'Month': [12, 11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1],
                'Interest_Rate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
                'Unemployment_Rate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
                'Stock_Index_Price': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]
                }

df = DataFrame(Stock_Market, columns=['Year', 'Month', 'Interest_Rate', 'Unemployment_Rate', 'Stock_Index_Price'])

"""
Checking for Linearity
Before you execute a linear regression model, it is advisable to validate that certain assumptions are met.

As noted earlier, you may want to check that a linear relationship exists between the dependent variable and 
the independent variable/s.

In our example, you may want to check that a linear relationship exists between:

The Stock_Index_Price (dependent variable) and the Interest_Rate (independent variable); and
The Stock_Index_Price (dependent variable) and the Unemployment_Rate (independent variable)
To perform a quick linearity check, you can use scatter diagrams (utilizing the matplotlib library):
"""

plt.scatter(x=df['Interest_Rate'], y=df['Stock_Index_Price'], color='red')
plt.title("Stock index price Vs Interest rate", fontsize=14)
plt.xlabel("Interest Rate", fontsize=14)
plt.ylabel("Stock index price", fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(x=df["Unemployment_Rate"], y=df['Stock_Index_Price'], color='green')
plt.title("Stock index price Vs Unemployment rate", fontsize=14)
plt.xlabel("Unemployment rate", fontsize=14)
plt.ylabel("Stock index price", fontsize=14)
plt.grid(True)
plt.show()

# Here we have 2 variable for multiple regression, If you just want to use one variable for simple linear regression
# then use X = df['Interest_Rate'] for example
# Alternatively, you may add additional variable within the brackets
X = df[['Interest_Rate', 'Unemployment_Rate']]
Y = df['Stock_Index_Price']

# with sklearn
regressor = LinearRegression()
regressor.fit(X=X, y=Y)
print('Intercept: ', regressor.intercept_)
print('Coefficient: ', regressor.coef_)

# prediction with sklearn
New_Interest_Rate = 2.75
New_Unemployment_Rate = 5.3
pred = regressor.predict([[New_Interest_Rate ,New_Unemployment_Rate]])
print('Predicted Stock Index Price: \n', pred)

# with statsmodels
X = sm.add_constant(X)  # adding a constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print_model = model.summary()
print(print_model)

"""
This output includes the intercept and coefficients. You can use this information to build the multiple linear regression equation as follows:

Stock_Index_Price = (Intercept) + (Interest_Rate coef)*X1 + (Unemployment_Rate coef)*X2

And once you plug the numbers:

Stock_Index_Price = (1798.4040) + (345.5401)*X1 + (-250.1466)*X2

Imagine that you want to predict the stock index price after you collected the following data:

Interest Rate = 2.75 (i.e., X1= 2.75)
Unemployment Rate = 5.3 (i.e., X2= 5.3)
If you plug that data into the regression equation, you’ll get the exact same predicted results as displayed in the second part:

Stock_Index_Price = (1798.4040) + (345.5401)*(2.75) + (-250.1466)*(5.3) = 1422.86


"""