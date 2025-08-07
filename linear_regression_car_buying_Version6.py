import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data from CSV
df = pd.read_csv('example_cars.csv')

# Data preprocessing
df = df.dropna()

# Features and target
X = df[['year', 'mileage', 'horsepower']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Predictions:", y_pred)
print("R^2 Score:", r2_score(y_test, y_pred))