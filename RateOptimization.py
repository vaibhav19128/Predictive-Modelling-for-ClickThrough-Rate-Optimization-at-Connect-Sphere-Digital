import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("advertising.csv")  # Replace with your dataset file name
print(df.head())

print(df.info())
print(df.describe())
sns.pairplot(df, hue='Clicked on Ad')
plt.show()

X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
y = df['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Example: Age=30, Time=80, Income=60000, Usage=200
new_user = np.array([[80, 30, 60000, 200]])
prediction = model.predict(new_user)
print("Predicted Click:", "Yes" if prediction[0] == 1 else "No")

