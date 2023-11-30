import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create a sample DataFrame
data = {'size': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
        'bedrooms': [3, 3, 2, 4, 2, 3, 4, 4, 3, 2],
        'location': ['A', 'B', 'A', 'C', 'B', 'C', 'C', 'B', 'A', 'A'],
        'price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]}

df = pd.DataFrame(data)

# Select feature and target variable
feature = df[['size']]  # Select the desired feature (e.g., house size)
target = df['price']

# Perform bivariate analysis (scatter plot)
plt.scatter(feature, target)
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.title('Bivariate Analysis: House Size vs. Price')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)

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
plt.scatter(X_test, y_test, label='Actual Prices')
plt.scatter(X_test, y_pred, label='Predicted Prices')
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()

# Get coefficient to understand feature importance
coefficients = pd.DataFrame({'Feature': feature.columns, 'Coefficient': model.coef_})
print(coefficients)
