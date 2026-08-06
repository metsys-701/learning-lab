"""
Topic: Model Evaluation Metrics

Description:
This file reuses the SVM model to practice calculating confusion
matrix, precision, recall, F1, and ROC-AUC instead of just accuracy.
"""

#=========================
#1. Imports
#=========================
# Import pandas, train_test_split, SVC, StandardScaler, and the evaluation metrics
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#=========================
#2. Load and Prepare Data
#=========================

df = pd.read_csv("../projects/01_titanic/train.csv")

df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin', 'Fare', 'Embarked'], inplace=True)
df.dropna(axis=0, how='any', inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

#=========================
#3-4. Split Features/Target, Train/Test Split
#=========================

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#=========================
#5. Feature Scaling
#=========================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#=========================
#6. Train SVM Model
#=========================
# Reuse SVM since it performed best

model = SVC(kernel='rbf', probability=True)
model.fit(X_train_scaled, y_train)


#=========================
#7. Evaluation Metrics
#=========================
# Calculate confusion matrix, precision, recall, F1, and ROC-AUC
predictions = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:", cm)
precision = precision_score(y_test, predictions)
print("Precision score:", precision)
recall = recall_score(y_test, predictions)
print("Recall Score:", recall)
f1 = f1_score(y_test, predictions)
print("F1:", f1)
auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
print("ROC AUC:", auc)

#=========================
#8. Result Note
#=========================
# Precision: ~74%, Recall: ~71%, F1: ~73%, ROC-AUC: ~83%
# Recall being a bit lower than precision means the model missed some
# passengers who actually survived (18 false negatives). Accuracy alone
# (~85% in earlier runs) didn't show this imbalance between precision
# and recall.