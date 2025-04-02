# logistic_regression_churn.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------
# 1️⃣ Load Data
# --------------------------------------------------------------------------------

# Dataset URL
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"

# Read CSV
df = pd.read_csv(url)

# Display first few rows
print("📌 First 5 rows of the dataset:")
print(df.head())

# --------------------------------------------------------------------------------
# 2️⃣ Exploratory Data Analysis (EDA)
# --------------------------------------------------------------------------------

print("\n🛠️ Dataset Information:")
print(df.info())  # Check for missing values and data types

print("\n🔍 Summary Statistics:")
print(df.describe())  # Summary statistics

# Visualizing churn distribution
plt.figure(figsize=(5, 3))
sns.countplot(x=df['churn'], palette='coolwarm')
plt.title("Churn Distribution")
plt.show()

# Checking feature correlations
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# --------------------------------------------------------------------------------
# 3️⃣ Data Preprocessing
# --------------------------------------------------------------------------------

# Selecting relevant features
features = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']
target = 'churn'

X = df[features].values  # Independent variables
y = df[target].astype(int).values  # Dependent variable (converted to integer)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------------------------------------
# 4️⃣ Train-Test Split
# --------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\n✅ Data split completed: Training samples:", len(X_train), "| Testing samples:", len(X_test))

# --------------------------------------------------------------------------------
# 5️⃣ Train Logistic Regression Model
# --------------------------------------------------------------------------------

LR = LogisticRegression()
LR.fit(X_train, y_train)

# Predict on test data
y_pred = LR.predict(X_test)
y_pred_proba = LR.predict_proba(X_test)

print("\n🎯 Predictions (First 10):", y_pred[:10])
print("\n🔢 Prediction Probabilities (First 10):\n", y_pred_proba[:10])

# --------------------------------------------------------------------------------
# 6️⃣ Performance Evaluation
# --------------------------------------------------------------------------------

# Log Loss
logloss_value = log_loss(y_test, y_pred_proba)
print("\n📉 Log Loss:", round(logloss_value, 4))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("✅ Accuracy Score:", round(accuracy * 100, 2), "%")

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(5, 3))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# --------------------------------------------------------------------------------
# 7️⃣ Baseline Model (Dummy Classifier for Comparison)
# --------------------------------------------------------------------------------

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_dummy_pred = dummy.predict(X_test)

dummy_accuracy = accuracy_score(y_test, y_dummy_pred)
print("\n🆚 Baseline Model Accuracy (Majority Class Prediction):", round(dummy_accuracy * 100, 2), "%")

# --------------------------------------------------------------------------------
# 8️⃣ Feature Importance Visualization
# --------------------------------------------------------------------------------

coefficients = pd.Series(LR.coef_[0], index=features)
coefficients.sort_values().plot(kind='barh', color='teal', figsize=(6, 4))
plt.title("🔎 Feature Importance in Logistic Regression")
plt.xlabel("Coefficient Value")
plt.show()

# --------------------------------------------------------------------------------
print("\n🎉 Model Training and Evaluation Completed!")
