"""
linear_regression_car_buying.py

A simple linear regression model for analyzing car buying trends.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def load_data(filepath):
    """
    Load car data from a CSV file.
    """
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Preprocess the data (drop missing values, etc.).
    """
    return df.dropna()

def train_model(X, y):
    """
    Train a linear regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Make predictions and evaluate the model performance.
    """
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print("Predictions:", y_pred)
    print("R^2 Score:", score)

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data('example_cars.csv')
    df = preprocess_data(df)
    
    # Features and target
    X = df[['year', 'mileage', 'horsepower']]
    y = df['price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)