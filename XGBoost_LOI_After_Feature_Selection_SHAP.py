"""
This script performs the following tasks:

Loads a dataset from a CSV file, splits it into features (X) and target variable (y), and applies standard scaling to the data.
Splits the data into training and testing sets, and then trains an XGBoost regression model.
Evaluates the model using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.
Loads a new PubMed dataset, scales it using the previously fitted scaler, and uses the trained XGBoost model to predict values for the new dataset.
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def load_and_preprocess_data(filepath):
    """Load data from a CSV file and split into features and target."""
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].values  # Features (all columns except the last one)
    y = df.iloc[:, -1].values   # Target variable (last column)
    return X, y

def scale_data(X_train, X_test):
    """Standardize the training and test data."""
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit scaler on the training data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_xgboost_model(X_train, y_train, params, num_round=100):
    """Train an XGBoost model with specified parameters."""
    dtrain = xgb.DMatrix(X_train, label=y_train)  # Convert to DMatrix
    model = xgb.train(params, dtrain, num_round)  # Train the model
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model using MSE, RMSE, and R² score."""
    dtest = xgb.DMatrix(X_test, label=y_test)  # Convert to DMatrix
    y_pred = model.predict(dtest)  # Make predictions
    mse = mean_squared_error(y_test, y_pred)  # Compute MSE
    rmse = mse ** 0.5  # Compute RMSE
    r2 = r2_score(y_test, y_pred)  # Compute R² score
    return mse, rmse, r2

def predict_on_new_data(model, X_new, scaler):
    """Apply the model to new data after scaling."""
    X_new_scaled = scaler.transform(X_new)  # Scale new data using the fitted scaler
    dnew = xgb.DMatrix(X_new_scaled)  # Convert to DMatrix
    y_new_pred = model.predict(dnew)  # Make predictions
    return y_new_pred

def main():
    # File paths for the datasets
    dataset_filepath = 'XGBOOST加数据归一化_TOP85_dataset.csv'
    pubmed_filepath = '20240612_SMILES_COMBINE_48w-valid_smiles_TOP85_dataset.csv'
    
    # Load and preprocess main dataset
    X, y = load_and_preprocess_data(dataset_filepath)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    # Initialize XGBoost parameters
    params = {
        'max_depth': 10,  # Maximum depth of trees
        'objective': 'reg:squarederror',  # Regression objective
        'eval_metric': 'rmse',  # RMSE for evaluation
        'colsample_bytree': 0.7,  # Subsample ratio for columns
        'eta': 0.1,  # Learning rate
        'subsample': 0.8,  # Subsample ratio for rows
    }
    
    # Train the XGBoost model
    model = train_xgboost_model(X_train_scaled, y_train, params)
    
    # Evaluate the model
    mse, rmse, r2 = evaluate_model(model, X_test_scaled, y_test)
    
    print(f"Model Evaluation:\nMSE: {mse}\nRMSE: {rmse}\nR²: {r2}")
    
    # Load and preprocess the PubMed dataset
    X_pubmed, y_pubmed = load_and_preprocess_data(pubmed_filepath)
    
    # Predict on new PubMed data
    y_pubmed_pred = predict_on_new_data(model, X_pubmed, scaler)
    
    # Output predictions
    print(f"Predictions on PubMed data: {y_pubmed_pred}")

if __name__ == "__main__":
    main()
