# Lamar Undergrad Research
# Name: Austin Robertson  
# Date: 10/01/2024

import torch
import torch.nn as nn 
import torch.optim as optim
import pandas as pd
from KaggleModelClass import SimpleNN

# Load the preprocessed data
X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
y_train = pd.read_csv('y_train.csv')
y_val = pd.read_csv('y_val.csv')

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use the CPU as I am using a RTX4070 which is CUDA capabable
input_size = X_train.shape[1] # Extracts number of features (columns) from the training data
model = SimpleNN(input_size).to(device) # Initialize our model object from the SimpleNN class

# Define loss function and optimizer with L2 regularization
criterion = nn.MSELoss() # Using mean square average for our loss function
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # weight_decay is the L2 regularization parameter

# Training loop
num_epochs = 25
for epoch in range(num_epochs): # Set our for loop for the model training
    model.train() # Set the model to training mode
    running_loss = 0.0 # Acummulate the losss over the current epoch. Helps track the loss after each epoch

    # Convert data to torch tensors
    inputs = torch.tensor(X_train.values, dtype=torch.float32).to(device) # Converts the training data to tensors
    labels = torch.tensor(y_train.values, dtype=torch.float32).to(device).squeeze() # Converts the labels for the training data to tensors

    # Zero the parameter gradients
    optimizer.zero_grad() # Gradients are accumulated by default in PyTorch so they need to be reset before backward pass

    # Forward pass
    outputs = model(inputs) # Perform a forward pass through the model
    # Compute the loss between the model's prediction outputs and the true labels
    loss = criterion(outputs, labels.unsqueeze(1)) # The .unsqueeze() function is make sure the labels tensors is the same shape as the outputs tensors

    # Backpropagation
    loss.backward() # Compute the gradient of the loss models weights
    optimizer.step() # Update the weights using the computed grandients

    running_loss += loss.item() # loss.item() extracts the scalar value of the loss tensor
    # Print the loss for each epoch
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(X_train)}') # Calculate the average total loss by dividing running_loss by the length of total number of training samples 

# Validation
model.eval() # This sets the model to evaluation mode
with torch.no_grad(): # Disable gradients since we won't be adjusting the weights
    val_inputs = torch.tensor(X_val.values, dtype=torch.float32).to(device) # Convert the validation data to tensors
    val_labels = torch.tensor(y_val.values, dtype=torch.float32).to(device).squeeze() # Convert the validation labels to tensors
    val_outputs = model(val_inputs) # Perform a forward pass through the model to get the models prediction after training
    val_loss = criterion(val_outputs, val_labels.unsqueeze(1)) # Compute the loss between the model's predictions and the labels
    print(f'Validation Loss: {val_loss.item()}') # Prints the validation loss for the current epoch

# print(f"Inputs tensor: {inputs}, Labels tensor: {labels}") # Print tensor structure

# Save the trained model
torch.save(model.state_dict(), 'deforestation_prediction_model.pth') # model.state_dict() contains the model's parameters

