"""
This script performs XGBoost regression on a dataset with varying numbers of features. For each subset of features (from 85 to 86), it:

Loads the dataset from a CSV file.
Splits the dataset into training and test sets.
Standardizes the features.
Trains an XGBoost model and evaluates its performance using metrics such as MSE (Mean Squared Error), RMSE (Root Mean Squared Error), and R².
Prints the performance metrics for each feature subset.
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filepath):
    """Load dataset from a CSV file and separate features (X) and target variable (y)."""
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1]  # All columns except the last one as features
    y = df.iloc[:, -1]   # Last column as the target variable
    return X, y

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and test sets, then standardize the features."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit and transform the training data
    X_test = scaler.transform(X_test)        # Transform the test data
    return X_train, X_test, y_train, y_test

def train_xgboost_model(X_train, y_train, params, num_round=100):
    """Train an XGBoost model with the given parameters."""
    dtrain = xgb.DMatrix(X_train, label=y_train)  # Convert to XGBoost's DMatrix format
    model = xgb.train(params, dtrain, num_round)  # Train the model
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using MSE, RMSE, and R² score."""
    dtest = xgb.DMatrix(X_test, label=y_test)  # Convert test set to DMatrix format
    y_pred = model.predict(dtest)  # Predict using the model
    mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
    rmse = mse ** 0.5  # Calculate Root Mean Squared Error
    r2 = r2_score(y_test, y_pred)  # Calculate R² score
    return mse, rmse, r2

def main():
    """Main function to execute the workflow of loading data, training, and evaluating."""
    filepath = 'XGBOOST加数据归一化_TOPX_dataset.csv'  # Path to the dataset

    # Load the data
    X, y = load_data(filepath)

    # Initialize a dictionary to store performance scores
    performance_scores = {}

    # Iterate over different subsets of features (85 features in this case)
    for num_features in range(85, 86, 1):  # Iterating over feature subsets
        X_subset = X.iloc[:, :num_features]  # Select the first `num_features` columns

        # Split and scale the data
        X_train, X_test, y_train, y_test = split_and_scale_data(X_subset, y)

        # XGBoost model parameters
        params = {
            'max_depth': 10,  # Maximum depth of the trees
            'objective': 'reg:squarederror',  # Regression task
            'eval_metric': 'rmse',  # Evaluation metric: Root Mean Squared Error
            'colsample_bytree': 0.7,
            'eta': 0.1,  # Learning rate
            'subsample': 0.8  # Subsampling ratio
        }

        # Train the model
        model = train_xgboost_model(X_train, y_train, params)

        # Evaluate the model
        mse, rmse, r2 = evaluate_model(model, X_test, y_test)

        # Store performance scores
        performance_scores[num_features] = {'MSE': mse, 'RMSE': rmse, 'R2': r2}

    # Print performance scores for each subset of features
    for num_features, scores in sorted(performance_scores.items()):
        print(f"Number of features: {num_features}, MSE: {scores['MSE']}, RMSE: {scores['RMSE']}, R2: {scores['R2']}")

if __name__ == "__main__":
    main()
