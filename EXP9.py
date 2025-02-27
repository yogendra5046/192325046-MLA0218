import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (Example: Diabetes dataset)
from sklearn.datasets import load_diabetes
data = load_diabetes()
X, y = data.data[:, np.newaxis, 2], data.target  # Using one feature for simplicity

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_linear = linear_reg.predict(X_test)

# Polynomial Regression model (degree = 2)
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)

# Evaluate models
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f'Linear Regression MSE: {mse_linear:.2f}, R^2: {r2_linear:.2f}')
print(f'Polynomial Regression MSE: {mse_poly:.2f}, R^2: {r2_poly:.2f}')

# Plot results
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.scatter(X_test, y_pred_linear, color='red', label='Linear Regression Predictions')
plt.scatter(X_test, y_pred_poly, color='green', label='Polynomial Regression Predictions')
plt.legend()
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear vs Polynomial Regression')
plt.show()
