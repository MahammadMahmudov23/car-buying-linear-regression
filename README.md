# Car Buying Linear Regression

A Python implementation of a linear regression model to analyze and predict trends in current top car purchases. This repository includes sample code, data structure examples, and step-by-step instructions for training and evaluating the model using scikit-learn.

## Features

- Simple, readable Python code using pandas and scikit-learn
- Example dataset (customizable for your own data)
- Step-by-step model training and prediction
- Easily extendable to more features or larger datasets

## Requirements

- Python 3.7+
- pandas
- scikit-learn

Install requirements with:

```bash
pip install pandas scikit-learn
```

## Usage

1. Clone this repository:

    ```bash
    git clone https://github.com/MahammadMahmudov23/car-buying-linear-regression.git
    cd car-buying-linear-regression
    ```

2. Run the example script:

    ```bash
    python linear_regression_car_buying.py
    ```

3. Customize the `data` dictionary in `linear_regression_car_buying.py` with your own car buying data as needed.

## Example

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Example data (replace with real car buying data)
data = {
    'year': [2018, 2019, 2020, 2021, 2022, 2023],
    'mileage': [30000, 25000, 20000, 15000, 10000, 5000],
    'horsepower': [150, 160, 170, 180, 190, 200],
    'price': [20000, 22000, 24000, 26000, 28000, 30000]
}
df = pd.DataFrame(data)

# Features and target variable
X = df[['year', 'mileage', 'horsepower']]
y = df['price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict prices for the test data
y_pred = model.predict(X_test)

# Print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predictions:", y_pred)
```

---

Feel free to open issues or pull requests for improvements!