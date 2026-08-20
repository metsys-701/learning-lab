"""
Topic: MLP (Multi-Layer Perceptron) with PyTorch

Description:
This file trains my first neural network using PyTorch on the Titanic
dataset, and compares its accuracy with the classic ML models I tried
earlier (Logistic Regression, Decision Tree, Random Forest, KNN, SVM,
Naive Bayes).
"""
#=========================
#1. Imports
#=========================
# Import pandas, torch, torch.nn, train_test_split, and StandardScaler

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#=========================
#2. Load and Prepare Data
#=========================
# Same Titanic feature engineering as before

df = pd.read_csv("../projects/01_titanic/train.csv")


df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin', 'Fare', 'Embarked'], inplace=True)
df.dropna(axis=0, how='any', inplace=True)
df['FamilySİze'] = df['SibSp'] + df['Parch'] +1
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

#=========================
#3. Split Features and Target
#=========================

X = df.drop(columns=['Survived'])
y = df['Survived']

#=========================
#4. Train/Test Split
#=========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#=========================
#5. Feature Scaling
#=========================
# Scale the features (neural networks train better with scaled data)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled =scaler.transform(X_test)

#=========================
#6. Convert to PyTorch Tensors
#=========================
# Convert numpy arrays into PyTorch tensors

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

print(X_train_tensor)
print(type(X_train_tensor))
print(y_train_tensor.shape)

#=========================
#7. Define the MLP Model
#=========================
# Define a simple neural network with one hidden layer

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 8)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

model = MLP(input_size=X_train_tensor.shape[1])

#=========================
#8. Loss Function and Optimizer
#=========================
# Binary cross-entropy loss for binary classification, Adam optimizer for updating weights

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#=========================
#9. Training Loop
#=========================
# Train the model for multiple epochs

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if(epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

#=========================
#10. Evaluate on Test Set
#=========================
# Check accuracy on the test set

with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predicted = (test_outputs >= 0.5).float()
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f"Test Accuracy: {accuracy.item():.4f}")

#=========================
#11. Result Note
#=========================
# Test Accuracy: ~82.5%
# Loss dropped steadily from 0.6351 to 0.4146 over 100 epochs, which
# showed the model was actually learning. This beat Logistic Regression
# (~81%) and KNN (~81%), close to Random Forest (~80%) and Naive Bayes
# (~80%), but still behind SVM (~85%). Makes sense with what I learned —
# on a small tabular dataset like this, deep learning doesn't
# necessarily beat classic ML, since its real strength shows up with
# bigger and more complex data.