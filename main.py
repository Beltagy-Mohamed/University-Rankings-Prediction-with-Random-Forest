import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load data
def load_data(filepath):
    return pd.read_csv(filepath)

# Data preprocessing
def preprocess_data(data):
    # Drop rows with missing target values
    data = data.dropna(subset=['2024 Rank'])

    # Convert target to numeric, handling errors
    data['2024 Rank'] = pd.to_numeric(data['2024 Rank'], errors='coerce')

    # Fill remaining missing values
    data.fillna(data.mean(numeric_only=True), inplace=True)

    # Convert categorical columns to numeric using Label Encoding
    label_encoders = {}
    categorical_columns = ['Institution Name', 'Location', 'Size']

    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    return data, label_encoders

# Feature and target separation
def split_features_target(data):
    X = data.drop(columns=['2024 Rank'])
    y = data['2024 Rank']
    return X, y

# Model training and hyperparameter optimization
def train_model(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

# Main function
def main():
    # File path to the dataset
    filepath = 'data/university_rankings.csv'

    # Load and preprocess data
    data = load_data(filepath)
    processed_data, label_encoders = preprocess_data(data)

    # Feature and target split
    X, y = split_features_target(processed_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model and perform hyperparameter optimization
    best_model, best_params, best_score = train_model(X_train, y_train)

    # Predict and evaluate the model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print results
    print("Best Parameters:", best_params)
    print("Best Score (R^2):", best_score)
    print(f'Random Forest Mean Squared Error: {mse}')
    print(f'Random Forest R^2 Score: {r2}')

if __name__ == "__main__":
    main()
