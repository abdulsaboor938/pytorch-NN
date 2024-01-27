import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import json

# Define the PyTorch neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 5)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.fc1(x))
        x = nn.functional.leaky_relu(self.fc2(x))
        x = nn.functional.leaky_relu(self.fc3(x))
        x = nn.functional.softmax(self.fc4(x), dim=1)
        return x

def init_train(data, dump, num_epochs=100):

    # Features and labels
    X = data.drop('y', axis=1).values
    y = data['y'].values

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # PyTorch data loaders
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        
    # Initialize the model, loss function, and optimizer
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.1)

    # Train the PyTorch model
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # convert all the model parameters to json
    import json
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.data.numpy().tolist()

    # save the model parameters to a json file
    with open(dump, 'w') as f:
        json.dump(params, f)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test, dtype=torch.float32))
        _, predicted = torch.max(test_outputs, 1)
        test_accuracy = accuracy_score(y_test, predicted.numpy())
        return('Test accuracy:', test_accuracy)
    
def re_train(data, checkpoint , dump, num_epochs=100):

    # Features and labels
    X = data.drop('y', axis=1).values
    y = data['y'].values

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # PyTorch data loaders
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        
    # Initialize the model, loss function, and optimizer
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.1)

    # load the model parameters from a json file
    with open(checkpoint, 'r') as f:
        params = json.load(f)

    # convert the model parameters from json to tensors
    for name, param in model.named_parameters():
        param.data = torch.tensor(params[name])

    # Train the PyTorch model
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # convert all the model parameters to json
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.data.numpy().tolist()

    # save the model parameters to a json file
    with open(dump, 'w') as f:
        json.dump(params, f)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test, dtype=torch.float32))
        _, predicted = torch.max(test_outputs, 1)
        test_accuracy = accuracy_score(y_test, predicted.numpy())
        return('Test accuracy:', test_accuracy)
    
def predict(data, checkpoint):
    
    # Features and labels
    X_test = data.drop('y', axis=1).values
    y_test = data['y'].values

    # Standardize the features
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
        
    # Initialize the model, loss function, and optimizer
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.1)

    # load the model parameters from a json file
    with open(checkpoint, 'r') as f:
        params = json.load(f)

    # convert the model parameters from json to tensors
    for name, param in model.named_parameters():
        param.data = torch.tensor(params[name])

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test, dtype=torch.float32))
        _, predicted = torch.max(test_outputs, 1)
        test_accuracy = accuracy_score(y_test, predicted.numpy())
        return('Test accuracy:', test_accuracy)