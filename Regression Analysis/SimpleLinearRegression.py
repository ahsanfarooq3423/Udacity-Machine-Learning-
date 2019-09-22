#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import os
os.getcwd()


data = pd.read_csv('Salary_Data.csv')


X = data.iloc[:,:-1].values
y = data.iloc[:,1].values




from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)

#Fitting Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)


#Visualizing the training set results
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salaries vs Experience')
plt.xlabel("Years of Experience (Training Set)")
plt.ylabel("Salary")
plt.show()

#Visualizing the test set results
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salaries vs Experience')
plt.xlabel("Years of Experience (Test Set)")
plt.ylabel("Salary")
plt.show()







