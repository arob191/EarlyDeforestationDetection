import torch
import torch.nn as nn

class SimpleNN(nn.Module): # Create a new class, inherit from nn.Module
    def __init__(self, input_size): 
        super(SimpleNN, self).__init__() # Call constructure of parent class
        self.fc1 = nn.Linear(input_size, 128) # Fully connected layer with input sized input features and 128 output features
        self.dropout1 = nn.Dropout(0.5) # Droput layer to help with overfitting
        self.fc2 = nn.Linear(128, 64) # Another fully connected layer
        self.dropout2 = nn.Dropout(0.5) # Another dropout layer
        self.fc3 = nn.Linear(64, 1) # Another fully connected layer

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Pass the input through the first layer
        x = self.dropout1(x) # Pass input through dropout layer
        x = torch.relu(self.fc2(x)) # Pass input through 3rd layer
        x = self.dropout2(x) # Pass input through second dropout layer
        x = self.fc3(x) # Pass input through fith layer
        return x # Return the model prediction