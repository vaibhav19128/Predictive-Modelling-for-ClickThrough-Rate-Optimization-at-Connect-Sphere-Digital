import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Load Dataset
df = pd.read_csv("advertising.csv")
print(" Dataset Loaded Successfully!\n")
print(df.head())
print("\nShape of dataset:", df.shape)
# Data Analysis
# --------------------------------------------
print("\n Checking for null values:")
print(df.isnull().sum())
print("\n Summary Statistics:")
print(df.describe())
# Feature Selection
X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
y = df['Clicked on Ad']
# Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)
# Model Evaluation
y_pred = model.predict(X_test)
print("\n Model Evaluation Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Predict for New User Example
# [Daily Time Spent on Site, Age, Area Income, Daily Internet Usage]
new_user = np.array([[80, 30, 60000, 200]])
prediction = model.predict(new_user)
print("\n Prediction for New User:")
print("Will Click on Ad?" , " Yes" if prediction[0] == 1 else " No")
# Visualization 
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()