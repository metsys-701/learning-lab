"""
Topic: Overfitting and Underfitting

Description:
This file trains a Decision Tree with different max_depth values
to compare train accuracy vs test accuracy, and see overfitting
happen as the tree gets deeper.
"""
#=========================
#1. Imports
#=========================
# Import pandas, train_test_split, DecisionTreeClassifier, and accuracy_score

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

#=========================
#2. Load and Prepare Data
#=========================
# Same Titanic feature engineering as before

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#=========================
#5. Compare Different max_depth Values
#=========================
# Train a Decision Tree with several different max_depth values,
# and compare train accuracy vs test accuracy for each

depths = [1, 2, 3, 5, 10, None]

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train,y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
   

    train_acc= accuracy_score(y_train, train_predictions)
    test_acc = accuracy_score(y_test, test_predictions)

    print(f"max_depth={depth} | Train: {train_acc}, Test: {test_acc}")


#=========================
#6. Result Note
#=========================
# max_depth=1: Train ~78%, Test ~79% (too simple, underfitting-ish)
# max_depth=3: Train ~83%, Test ~83% (closest gap, best balance)
# max_depth=10: Train ~92%, Test ~80% (big gap starts)
# max_depth=None: Train ~94%, Test ~82% (biggest gap, clear overfitting)
# Higher train accuracy didn't mean a better model. max_depth=3 had a
# much smaller train/test gap than max_depth=10 or None, even though
# those had higher train accuracy. This really showed me why test
# accuracy matters more than train accuracy.