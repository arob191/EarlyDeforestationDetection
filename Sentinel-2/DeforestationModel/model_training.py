import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler  # Updated AMP imports
from data_preparation import prepare_data
from model_definition import ResNet34MultiTask  # Using ResNet-34 with multitask heads
import matplotlib.pyplot as plt
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# Training function
def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=20, accumulation_steps=4):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Training loop
        optimizer.zero_grad()
        for i, (inputs, class_targets, reg_targets) in enumerate(train_loader):
            inputs, class_targets, reg_targets = inputs.to(device), class_targets.to(device), reg_targets.to(device)

            with autocast("cuda"):  # Updated autocast
                class_outputs, reg_outputs = model(inputs)  # Forward pass
                class_loss = classification_criterion(class_outputs, class_targets)  # Classification loss
                reg_loss = regression_criterion(reg_outputs, reg_targets)  # Regression loss
                loss = 0.5 * class_loss + 0.5 * reg_loss  # Combined loss
                loss = loss / accumulation_steps  # Divide by accumulation steps

            scaler.scale(loss).backward()  # Backpropagation with scaling

            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, class_targets, reg_targets in val_loader:
                inputs, class_targets, reg_targets = inputs.to(device), class_targets.to(device), reg_targets.to(device)

                with autocast("cuda"):  # Updated autocast
                    class_outputs, reg_outputs = model(inputs)
                    class_loss = classification_criterion(class_outputs, class_targets)
                    reg_loss = regression_criterion(reg_outputs, reg_targets)
                    val_loss += 0.5 * class_loss.item() + 0.5 * reg_loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Adjust learning rate
        scheduler.step()

        # Print losses
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    plot_losses(train_losses, val_losses)

# Plot training and validation losses
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

# Evaluate the model on the test set
def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, class_targets, reg_targets in test_loader:
            inputs, class_targets, reg_targets = inputs.to(device), class_targets.to(device), reg_targets.to(device)

            with autocast("cuda"):  # Updated autocast
                class_outputs, reg_outputs = model(inputs)
                class_loss = classification_criterion(class_outputs, class_targets)
                reg_loss = regression_criterion(reg_outputs, reg_targets)
                test_loss += 0.5 * class_loss.item() + 0.5 * reg_loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # File paths
    fazenda_csvs = [
        "E:/Sentinelv3/Fazenda Forest/Fazenda_2015_2016.csv",
        "E:/Sentinelv3/Fazenda Forest/Fazenda_2017_2018.csv",
        "E:/Sentinelv3/Fazenda Forest/Fazenda_2019_2020.csv",
    ]
    deforestation_csv = "E:/Sentinelv3/Fazenda Forest/deforestation_data.csv"

    # Prepare the data
    train_loader, val_loader, test_loader = prepare_data(fazenda_csvs, deforestation_csv, batch_size=16)  # Smaller batch size

    # Load the model
    model = ResNet34MultiTask(num_classes=1).to(device)

    # Define loss functions
    classification_criterion = nn.BCEWithLogitsLoss()  # For binary classification
    regression_criterion = nn.MSELoss()  # For NDVI regression

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Cyclic learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode="triangular2")

    # Mixed precision training setup
    scaler = GradScaler()

    # Train the model
    train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=20, accumulation_steps=4)

    # Save the trained model
    torch.save(model.state_dict(), "E:/Models/deforestation_model_resnet34_multitask.pth")
    print("Model saved to E:/Models/deforestation_model_resnet34_multitask.pth")

    # Evaluate the model
    evaluate_model(model, test_loader)

