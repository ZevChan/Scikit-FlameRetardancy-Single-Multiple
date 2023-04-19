# -*- coding: utf-8 -*-
"""
@author: Zhongwei Chen
Email: czw1995@outlook.com
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge

from sklearn.feature_selection import RFE

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Databse
dataset = pd.read_csv("TTI_Database.csv")
#dataset = pd.read_csv("LOI_Database.csv")
#dataset = pd.read_csv("pHRR_Database.csv")
#dataset = pd.read_csv("THR_Database.csv")
#dataset = pd.read_csv("94_Database.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset_BDOPO_OH = pd.read_csv("BDOPO_Database.csv")
X_BDOPO_OH = dataset_BDOPO_OH.iloc[:, :-4].values
y_BDOPO_OH = dataset_BDOPO_OH.iloc[:, -4].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)
# LOI, X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=80)
# PHRR, X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=82)
# THR, X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# 94, X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=76)

# Data processing
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_BDOPO_OH = scaler.transform(X_BDOPO_OH)

# Feature Selection
estimator = Ridge(alpha=10, random_state=1, max_iter=100000)
# LOI, estimator = Ridge(alpha=1, random_state=1, max_iter=100000)
# PHRR, estimator = Ridge(alpha=1, random_state=1, max_iter=100000)
# THR, estimator = Ridge(alpha=1, random_state=1, max_iter=100000)
# 94, estimator = Ridge(alpha=1, random_state=1, max_iter=100000)
selector = RFE(estimator, n_features_to_select = 153)
# LOI, selector = RFE(estimator, n_features_to_select = 282)
# PHRR, selector = RFE(estimator, n_features_to_select = 203)
# THR, selector = RFE(estimator, n_features_to_select = 25)
# 94, selector = RFE(estimator, n_features_to_select = 13)
selector.fit(X_train, y_train)
X_train_selector = selector.transform(X_train)
X_test_selector =selector.transform(X_test) 
X_BDOPO_OH_selector = selector.transform(X_BDOPO_OH)
                 
# Output Selected Feature
feature_selected = selector.get_support()
logf = open("logfile.log", "a+")
np.set_printoptions(threshold=np.inf)
print(f"{feature_selected}", file=logf, flush=True)

# Regression
regressor = GradientBoostingRegressor(n_estimators=210, max_depth=4, random_state=1)
# LOI, regressor = GradientBoostingRegressor(n_estimators=910, max_depth=4, random_state=1)
# PHRR, regressor = GradientBoostingRegressor(n_estimators=410, max_depth=3, random_state=1)
# THR, regressor = GradientBoostingRegressor(n_estimators=210, max_depth=4, random_state=1)
# 94, regressor = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=1)
 
regressor.fit(X_train_selector, y_train)

# Output feature importance
feature_importances_ = regressor.feature_importances_

# Model Performance
# LOI, PHRR, THR, TII
y_train_predict = regressor.predict(X_train_selector)
y_predict = regressor.predict(X_test_selector)
mean_squared_error = mean_squared_error(y_test, y_predict)
root_mean_squard_error = mean_squared_error**0.5
mean_absolute_error = mean_absolute_error(y_test, y_predict)
# 94
#accuracy_score_train = accuracy_score(y_train, y_train_predict)
#accuracy_score_test = accuracy_score(y_test, y_predict)
#precision_score = precision_score(y_test, y_predict, average='weighted')
#recall_score = recall_score(y_test, y_predict, average='weighted')
#f1_score = f1_score(y_test, y_predict, average='weighted')
                                                                        
print(f"train R2: {regressor.score(X_train_selector, y_train):.3f}")
print(f"test R2: {regressor.score(X_test_selector, y_test):.3f}")

# Predict EP/BDOPO
y_BDOPO_OH_predict = regressor.predict(X_BDOPO_OH_selector)