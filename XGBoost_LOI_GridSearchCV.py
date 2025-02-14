"""
This script performs hyperparameter tuning for an XGBoost regression model using GridSearchCV to optimize the model's performance. It:

Loads a dataset from a CSV file.
Splits the data into training and test sets, and applies standard scaling to the features.
Uses GridSearchCV to find the best hyperparameters for an XGBoost regressor by testing a range of parameters.
Evaluates the performance of the best model using the R² score on the test set.
"""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def load_data(filepath):
    """Load data from a CSV file and separate features and target variables."""
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1]  # Features (all columns except the last one)
    y = df.iloc[:, -1]   # Target variable (last column)
    return X, y

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and test sets, and apply standard scaling."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def perform_grid_search(X_train, y_train, param_grid, cv=10, n_jobs=-1):
    """Perform GridSearchCV to optimize model hyperparameters."""
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, 
                               scoring='r2', cv=cv, verbose=1, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_model(best_model, X_test, y_test):
    """Evaluate the best model using R² score on the test set."""
    y_pred = best_model.predict(X_test)
    r2_val = r2_score(y_test, y_pred)
    print(f"R² score on the test set: {r2_val}")
    return r2_val

def main():
    filepath = 'EP+FR+Curing_Dataset.csv'
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],  # Range of tree depths to test
        'eta': [0.01, 0.05, 0.1, 0.2, 0.3],      # Learning rate values to test
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Subsampling ratios
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Feature subsampling ratios
    }

    # Step 1: Load data
    X, y = load_data(filepath)

    # Step 2: Split and standardize the data
    X_train, X_test, y_train, y_test = split_and_scale_data(X, y)

    # Step 3: Perform hyperparameter tuning
    grid_search = perform_grid_search(X_train, y_train, param_grid)

    # Step 4: Print the best parameters found
    print("Best parameters set found on development set:")
    print(grid_search.best_params_)

    # Step 5: Evaluate the model with the best parameters
    best_model = grid_search.best_estimator_
    evaluate_model(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
