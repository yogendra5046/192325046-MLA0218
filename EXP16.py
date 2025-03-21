import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features for models like KNN and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary of models
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel='linear'),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naïve Bayes": GaussianNB()
}

# Evaluate models
results = {}
for name, model in models.items():
    if name in ["K-Nearest Neighbors", "Support Vector Machine"]:  # Use scaled data
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"\n{name} Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Convert results to a DataFrame for visualization
results_df = pd.DataFrame(list(results.items()), columns=["Algorithm", "Accuracy"])
results_df = results_df.sort_values(by="Accuracy", ascending=False)

# Plot accuracy comparison
plt.figure(figsize=(10, 5))
sns.barplot(x="Accuracy", y="Algorithm", data=results_df, palette="coolwarm")
plt.xlabel("Accuracy Score")
plt.ylabel("Classification Algorithm")
plt.title("Comparison of Classification Algorithms on the Iris Dataset")
plt.xlim(0.85, 1.0)  # Zoom into relevant range
plt.show()
