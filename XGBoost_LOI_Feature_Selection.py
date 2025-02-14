
"""
This script performs the following tasks:
1. Loads the dataset and separates features and the target variable.
2. Splits the dataset into training and test sets.
3. Scales the features using StandardScaler.
4. Trains an XGBoost regression model.
5. Evaluates the model's performance using RMSE and R².
6. Analyzes and prints the feature importance.
"""

import pandas as pd  
import xgboost as xgb  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler    
from sklearn.metrics import mean_squared_error, r2_score  
import numpy as np  

def load_data(filepath):
    """Load dataset from CSV and separate features and target variable."""
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1]  # Features (all columns except the last one)
    y = df.iloc[:, -1]   # Target variable (last column)
    return X, y

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and test sets, and apply StandardScaler."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit scaler on the training data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_xgb_model(X_train, y_train, param, num_round=100):
    """Train an XGBoost regression model."""
    dtrain = xgb.DMatrix(X_train, label=y_train)  # Convert to DMatrix format
    bst = xgb.train(param, dtrain, num_round)  # Train the model
    return bst

def evaluate_model(bst, X_test, y_test):
    """Evaluate the trained model using RMSE and R² metrics."""
    dtest = xgb.DMatrix(X_test, label=y_test)  # Convert to DMatrix format
    preds = bst.predict(dtest)  # Make predictions on the test set
    
    # Calculate RMSE and R²
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print("Root Mean Squared Error: %.4f" % rmse)
    print("R² (coefficient of determination): %.4f" % r2)
    return rmse, r2

def analyze_feature_importance(bst):
    """Analyze and print feature importance."""
    importance = bst.get_fscore()  # Get feature importance scores
    print("Feature importance:")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {score}")

def main():
    filepath = 'EP+FR+Curing_Dataset.csv'
    
    # 1. Load data
    X, y = load_data(filepath)
    
    # 2. Split and scale data
    X_train, X_test, y_train, y_test = split_and_scale_data(X, y)
    
    # 3. Set XGBoost hyperparameters
    param = {
        'max_depth': 10,  # Tree depth range (3-10)
        'objective': 'reg:squarederror',  # Regression task
        'eval_metric': 'rmse',  # Evaluation metric: RMSE
        'colsample_bytree': 0.7,  # Feature subsampling ratio
        'eta': 0.1,  # Learning rate
        'subsample': 0.8,  # Data subsampling ratio
    }
    
    # 4. Train the model
    bst = train_xgb_model(X_train, y_train, param, num_round=100)
    
    # 5. Evaluate the model
    evaluate_model(bst, X_test, y_test)
    
    # 6. Analyze feature importance
    analyze_feature_importance(bst)

if __name__ == "__main__":
    main()
