import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data from CSV
df = pd.read_csv('example_cars.csv')

# Data preprocessing
df = df.dropna()

# Features and target
X = df[['year', 'mileage', 'horsepower']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

print("Model trained.")