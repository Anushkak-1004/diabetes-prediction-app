# model_training.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import matplotlib.pyplot as plt

# Create plots directory if it doesn't exist
os.makedirs('D:/Disease prediction project/plots/', exist_ok=True)

# Load dataset
df = pd.read_csv('D:/Disease prediction project/dataset/diabetes.csv')

# Prepare features and labels
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# Confusion matrix - plot and save
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
fig_cm, ax_cm = plt.subplots()
disp.plot(cmap='Blues', ax=ax_cm)
plt.title("Confusion Matrix")
plt.savefig('D:/Disease prediction project/plots/confusion_matrix.png')
plt.show()

# Feature importance - plot and save
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='teal')
plt.xlabel('Feature Importance')
plt.title('Random Forest - Feature Importance')
plt.tight_layout()
plt.savefig('D:/Disease prediction project/plots/feature_importance.png')
plt.show()

# Save model and scaler
with open('D:/Disease prediction project/model/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('D:/Disease prediction project/model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")

