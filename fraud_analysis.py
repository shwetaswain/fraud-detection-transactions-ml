# === Setup ===
import matplotlib
matplotlib.use('TkAgg')  # Ensures plots show in VS Code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# === Load Dataset ===
df = pd.read_csv("Transactions.csv")
print("✅ First 5 Rows of Data:")
print(df.head())

# === Data Preprocessing ===
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['type'], drop_first=True)

# === Split Features & Labels ===
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# === Train-Test Split with Stratification ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

print("\n✅ X_train shape:", X_train.shape)
print("✅ X_test shape:", X_test.shape)
print("\n✅ y_train class distribution:\n", y_train.value_counts(normalize=True))
print("\n✅ y_test class distribution:\n", y_test.value_counts(normalize=True))

# === Apply SMOTE to Balance the Classes ===
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\n✅ After SMOTE - Class distribution:\n", y_train_res.value_counts())

# === Train Random Forest Classifier ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# === Predict on Test Set ===
y_pred = model.predict(X_test)

# === Evaluation ===
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix (Text)
cm = confusion_matrix(y_test, y_pred)
print("\n✅ Confusion Matrix (Raw Counts):")
print(cm)

# === Plot Confusion Matrix Heatmap ===
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('📊 Confusion Matrix Heatmap')
plt.tight_layout()
plt.show()

# === Plot Feature Importances ===
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices], palette='viridis')
plt.title('📈 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()






