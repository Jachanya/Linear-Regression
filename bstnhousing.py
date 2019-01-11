#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing my datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_train = train.iloc[:, 1:-1]
y_train = train.iloc[:, -1]


X_test = test.iloc[:, 1:]
y_test = test.iloc[:, -1]


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#analyzing test result
import statsmodels.formula.api as sm
X_train = np.append(arr = np.ones((333,1)).astype(int), values = X_train, axis=1)
X_train = X_train[:,8]
X_opt = X_train
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()

