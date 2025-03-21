import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/anikannal/stock_market_prediction/master/MobilePriceData.csv"
df = pd.read_csv(url)

# Step 2: Display dataset information
print("Dataset Sample:")
print(df.head())

# Step 3: Check for missing values
print("\nMissing values in dataset:", df.isnull().sum().sum())

# Step 4: Define features (X) and target (y)
X = df.drop("price_range", axis=1)  # All features except price_range
y = df["price_range"]  # Target variable (0: Low, 1: Medium, 2: High, 3: Premium)

# Step 5: Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 7: Predict on test data
y_pred = rf_model.predict(X_test)

# Step 8: Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d", xticklabels=["Low", "Medium", "High", "Premium"], yticklabels=["Low", "Medium", "High", "Premium"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 9: Test with a sample mobile specification
sample_mobile = np.array([[5, 1, 500, 2000, 3, 0, 4, 2500, 15, 20, 100, 1, 1, 1, 8, 200, 1]])  # Example features
predicted_price = rf_model.predict(sample_mobile)
print(f"\nPredicted Price Range: {predicted_price[0]}")
