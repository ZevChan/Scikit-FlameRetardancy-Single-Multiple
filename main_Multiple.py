# -*- coding: utf-8 -*-
"""
@author: Zhongwei Chen
Email: czw1995@outlook.com
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.feature_selection import RFE

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score 

from sklearn.multioutput import MultiOutputRegressor

# Databse
dataset = pd.read_csv("LOI+TTI+PHRR+THR_Database.csv")
X = dataset.iloc[:, :-4].values
y = dataset.iloc[:, [285,286,287,288]].values

dataset_BDOPO_OH = pd.read_csv("BDOPO_Database.csv")
X_BDOPO_OH = dataset_BDOPO_OH.iloc[:, :-4].values
y_BDOPO_OH = dataset_BDOPO_OH.iloc[:, [285,286,287,288]].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)

# Data processing
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_BDOPO_OH = scaler.transform(X_BDOPO_OH)

scaler = StandardScaler()
scaler.fit(y_train)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)
y_BDOPO_OH = scaler.transform(y_BDOPO_OH)

# Feature Selection
estimator = Lasso(alpha=1, random_state=1, max_iter=100000)
#random_state=1, max_iter=100000
selector = RFE(estimator, n_features_to_select=274)
#selector = RFECV(estimator, cv=cv)
selector.fit(X_train, y_train)
X_train_selector = selector.transform(X_train)
#print(f"X_train.shape :{X_train.shape}")
X_test_selector =selector.transform(X_test)
#print(f"X_test.shape :{X_test.shape}")
X_BDOPO_OH_selector = selector.transform(X_BDOPO_OH)
X_10RING_selector = selector.transform(X_10RING)

# Output Selected Feature
feature_selected = selector.get_support()
logf = open("logfile.log", "a+")
np.set_printoptions(threshold=np.inf)
print(f"{feature_selected}", file=logf, flush=True)

# Regression
regressor = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=410,
                                                           max_depth=3,
                                                           random_state=1))

regressor.fit(X_train_selector, y_train)

# Model Performance
y_train_predict = regressor.predict(X_train_selector)
y_predict = regressor.predict(X_test_selector)  
mean_squared_error = mean_squared_error(y_test, y_predict)
root_mean_squard_error = mean_squared_error**0.5
mean_absolute_error = mean_absolute_error(y_test, y_predict)

print(f"features={features} estimator={estimator} regressor={regressor} train R2: {regressor.score(X_train_selector, y_train):.3f}")
print(f"features={features} estimator={estimator} regressor={regressor} test R2: {regressor.score(X_test_selector, y_test):.3f}")

y_train1 = scaler.inverse_transform(y_train)
y_test1 = scaler.inverse_transform(y_test)
y_train_predict1 = scaler.inverse_transform(y_train_predict)
y_predict1 = scaler.inverse_transform(y_predict)

# Predict EP/BDOPO
y_BDOPO_OH_predict1 = regressor.predict(X_BDOPO_OH_selector)
y_BDOPO_OH_predict = scaler.inverse_transform(y_BDOPO_OH_predict1)