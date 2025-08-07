import pandas as pd

# Load data from CSV
df = pd.read_csv('example_cars.csv')

# Data preprocessing (if needed, e.g., handle missing values)
df = df.dropna()

print("Data after preprocessing:")
print(df.head())