"""
Topic: Hyperparameter Tuning

Description:
This file uses GridSearchCV on Random Forest to automatically try
different combinations of n_estimators and max_depth, instead of
manually testing values one by one.
"""
#=========================
#1. Imports
#=========================
# Import pandas, train_test_split, RandomForestClassifier, GridSearchCV, and accuracy_score

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#=========================
#2. Load and Prepare Data
#=========================
# Same Titanic feature engineering as before

df = pd.read_csv("../projects/01_titanic/train.csv")

df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin', 'Fare', 'Embarked'], inplace=True)
df.dropna(axis=0, how='any', inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Sex'] = df['Sex'].map({'male':0, 'female': 1})


#=========================
#3-4. Split Features/Target, Train/Test Split
#=========================

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#=========================
#5. Define Parameter Grid
#=========================
# Define the combinations of parameters to try

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10]
}

#=========================
#6. Grid Search
#=========================
# Try every combination and find the best one using cross-validation

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train,y_train)

#=========================
#7. Best Parameters and Score
#=========================
# Print the best parameters and the best cross-validation score

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

#=========================
#8. Evaluate on Test Set
#=========================
# Use the best model to predict on the test set

prediction = grid_search.predict(X_test)
print("predict:", prediction)

Acc = accuracy_score(y_test, prediction)
print("Accuracy score:", Acc)

#=========================
#9. Result Note
#=========================
# Best params: max_depth=3, n_estimators=50, CV score: ~82.5%
# Test accuracy: ~79%
# Interesting that the best combination wasn't the biggest one
# (n_estimators=200, max_depth=10) — same lesson as the overfitting
# experiment, bigger isn't always better. The CV score and test
# accuracy weren't identical, which makes sense since CV is an
# average over several splits, not the same as one specific test set.