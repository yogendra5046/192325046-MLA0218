import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Dataset
data = {
    'Age': [25, 45, 35, 50, 23, 40, 60, 30, 27, 55],
    'Income': [50000, 80000, 60000, 120000, 30000, 90000, 200000, 70000, 40000, 150000],
    'Credit_Score': [700, 650, 720, 680, 600, 750, 780, 640, 670, 770],
    'Loan_Amount': [20000, 50000, 25000, 80000, 15000, 60000, 100000, 30000, 18000, 70000],
    'Loan_Status': ['Approved', 'Rejected', 'Approved', 'Approved', 'Rejected',
                    'Approved', 'Approved', 'Rejected', 'Rejected', 'Approved']
}

df = pd.DataFrame(data)
print("Dataset Sample:\n", df.head())

# Step 2: Preprocess Data
le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])  # Convert Approved/Rejected to 1/0

# Step 3: Define Features (X) and Target (y)
X = df.drop("Loan_Status", axis=1)  # Features
y = df["Loan_Status"]  # Target Variable

# Step 4: Split Dataset into Train (80%) & Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize Features (GaussianNB Assumes Normal Distribution)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Naïve Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# Step 7: Make Predictions
y_pred = nb_model.predict(X_test_scaled)

# Step 8: Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nNaïve Bayes Model Accuracy: {accuracy:.2f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d",
            xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naïve Bayes")
plt.show()

# Step 9: Predict Loan Approval for a New Customer
new_customer = np.array([[28, 45000, 680, 20000]])  # Sample input: Age, Income, Credit Score, Loan Amount
new_customer_scaled = scaler.transform(new_customer)
prediction = nb_model.predict(new_customer_scaled)
print(f"\nPredicted Loan Status: {'Approved' if prediction[0] == 1 else 'Rejected'}")
