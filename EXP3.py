import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
# Create the dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Encode categorical values into numerical values
label_encoders = {}
for column in df.columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Split data into features (X) and target (y)
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target variable
# Create and train the Decision Tree classifier
clf = DecisionTreeClassifier(criterion='entropy')  # Using ID3 (entropy-based)
clf.fit(X, y)

# Display the tree structure
tree_rules = export_text(clf, feature_names=list(df.columns[:-1]))
print("Decision Tree Structure:\n", tree_rules)
# Encode the new sample using label encoders
sample = pd.DataFrame([['Sunny', 'Cool', 'High', 'Strong']], columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])
for column in sample.columns:
    sample[column] = label_encoders[column].transform(sample[column])

# Predict the class
prediction = clf.predict(sample)
predicted_label = label_encoders['PlayTennis'].inverse_transform(prediction)[0]

print("\nNew Sample Classification:")
print(f"Predicted Class: {predicted_label}")