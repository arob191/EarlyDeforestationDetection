# Lamar Undergrad Research
# Name: Austin Robertson  
# Date: 10/01/2024
# Design and train a CNN to detect future deforestation hotspots

import torch
import torch.nn as nn  # Add this import statement
import torch.optim as optim
import pandas as pd
from KaggleModelClass import SimpleNN

# # # Prepare your data transformations
# csv_file_path = 'E:\labels.csv'
# transform = None
# dataset = ds.DeforestationDataset(csv_file=csv_file_path, transform=transform)

# # Check to see if torch.cuda or torhc.backends.mps are available
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")


# # Test convertion to tensor
# for i in range(len(dataset)):
#     before_image, after_image, label = dataset[i]
#     print("Before image shape:", before_image.shape)
#     print("After image shape:", after_image.shape)
#     print("Label:", label)

# # Training loop

# # Initialize the model, loss function, and optimizer

# Load the preprocessed data
X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
y_train = pd.read_csv('y_train.csv')
y_val = pd.read_csv('y_val.csv')

# Convert data to torch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = X_train.shape[1]
model = SimpleNN(input_size).to(device)

# Define loss function and optimizer with L2 regularization
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # weight_decay is the L2 regularization parameter

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Convert data to torch tensors
    inputs = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    labels = torch.tensor(y_train.values, dtype=torch.float32).to(device).squeeze()

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels.unsqueeze(1))

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(X_train)}')

# Validation
model.eval()
with torch.no_grad():
    val_inputs = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    val_labels = torch.tensor(y_val.values, dtype=torch.float32).to(device).squeeze()
    val_outputs = model(val_inputs)
    val_loss = criterion(val_outputs, val_labels.unsqueeze(1))
    print(f'Validation Loss: {val_loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'resnet50_deforestation_model.pth')

