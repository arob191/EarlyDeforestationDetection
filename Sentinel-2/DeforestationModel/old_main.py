import torch
import torch.nn as nn
import torch.optim as optim
from old_data_preparation import prepare_data
from old_model_definition import get_resnet101_model  # Import the function
import matplotlib.pyplot as plt

# Check device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. File paths for the CSV files
fazenda_csvs = [
    "E:/Sentinelv3/Fazenda Forest/Fazenda_2015_2016.csv",
    "E:/Sentinelv3/Fazenda Forest/Fazenda_2017_2018.csv",
    "E:/Sentinelv3/Fazenda Forest/Fazenda_2019_2020.csv"
]
deforestation_csv = "E:/Sentinelv3/Fazenda Forest/deforestation_data.csv"  # Path to deforestation data CSV

# 2. Prepare the data
train_loader, val_loader, test_loader = prepare_data(fazenda_csvs, deforestation_csv, batch_size=32)

# 3. Load the ResNet-101 model
model = get_resnet101_model().to(device)

# 4. Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=8):
        self.patience = patience
        self.counter = 0
        self.best_val_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopping = EarlyStopping(patience=8)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, epochs=20):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step()
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    plot_losses(train_losses, val_losses)

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

# Save the model
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, epochs=20)
save_model(model, "C:/Users/Austin/OneDrive/Documents/Personal Projects/GitHub/EarlyDeforestationDetection/deforestation_model_resnet101.pth")

# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

evaluate_model(model, test_loader)