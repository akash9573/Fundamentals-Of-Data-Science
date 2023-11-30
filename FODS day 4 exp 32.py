import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Generating synthetic data
np.random.seed(42)
num_samples = 1000
engine_size = np.random.uniform(1.0, 5.0, num_samples)
horsepower = np.random.uniform(80, 300, num_samples)
fuel_efficiency = np.random.uniform(10, 50, num_samples)
price = 10000 + 2000 * engine_size + 30 * horsepower - 20 * fuel_efficiency + np.random.normal(0, 5000, num_samples)

# Creating a DataFrame
df = pd.DataFrame({'engine_size': engine_size, 'horsepower': horsepower, 'fuel_efficiency': fuel_efficiency, 'price': price})

# Select features and target variable
features = df[['engine_size', 'horsepower', 'fuel_efficiency']]
target = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()

# Get coefficients to understand feature importance
coefficients = pd.DataFrame({'Feature': features.columns, 'Coefficient': model.coef_})
print(coefficients)
