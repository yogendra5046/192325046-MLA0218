import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Generate Sample Sales Data (or load real-world data)
np.random.seed(42)
data = {
    'Month': np.arange(1, 13),  # Months 1-12
    'Ad_Spend': np.random.randint(1000, 5000, 12),  # Advertisement Spend
    'Discount': np.random.randint(5, 25, 12),  # Discount Percentage
    'Store_Visitors': np.random.randint(500, 3000, 12),  # Monthly Footfall
    'Sales': np.random.randint(10000, 50000, 12)  # Sales Revenue
}

df = pd.DataFrame(data)
print("Sample Sales Data:\n", df.head())

# Step 2: Define Features (X) and Target (y)
X = df[['Month', 'Ad_Spend', 'Discount', 'Store_Visitors']]
y = df['Sales']

# Step 3: Split Data into Train (80%) & Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the Features (for better accuracy)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate Model Performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 8: Predict Future Sales for Next Month
next_month_data = np.array([[13, 4500, 15, 2800]])  # Future values for Month=13
next_month_scaled = scaler.transform(next_month_data)
future_sales_pred = model.predict(next_month_scaled)
print(f"\nPredicted Sales for Next Month: ${future_sales_pred[0]:,.2f}")

# Step 9: Plot Actual vs Predicted Sales
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
