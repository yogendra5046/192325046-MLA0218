import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Generate Synthetic Dataset
np.random.seed(42)
data_size = 1000
df = pd.DataFrame({
    'income': np.random.randint(20000, 150000, data_size),
    'debt': np.random.randint(1000, 50000, data_size),
    'loan_amount': np.random.randint(5000, 100000, data_size),
    'payment_history': np.random.choice([0, 1], data_size, p=[0.3, 0.7]),  # 0: Bad, 1: Good
    'credit_score': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], data_size, p=[0.2, 0.3, 0.3, 0.2])
})

# Step 2: Preprocessing
label_encoder = LabelEncoder()
df['credit_score'] = label_encoder.fit_transform(df['credit_score'])  # Encode labels
X = df.drop(columns=['credit_score'])
y = df['credit_score']

# Normalize numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# Step 6: Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Confusion Matrix Visualization
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Credit Score Classification")
plt.show()
