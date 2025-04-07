import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
import torch.nn.functional as F
from torch.amp import autocast
from data_preparation import prepare_data
from model_definition import ResNet50MultiTask
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Focal Loss for multi-class classification
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Cross-Entropy Loss for multi-class classification
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)  # Probability of the predicted class
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        return F_loss.mean()

def plot_losses(train_losses, val_losses):
    """
    Plots training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=20, accumulation_steps=4):
    """
    Trains the multi-task learning model.
    """
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        optimizer.zero_grad()
        for i, (inputs, class_targets, reg_targets) in enumerate(train_loader):
            inputs, class_targets, reg_targets = inputs.to(device), class_targets.to(device), reg_targets.to(device)

            # Ensure classification targets are in the correct format
            class_targets = class_targets.squeeze(1).long()  # Convert to integer indices

            with autocast(device_type="cuda", enabled=True):
                class_outputs, reg_outputs = model(inputs)
                class_loss = classification_criterion(class_outputs, class_targets)  # Multi-class classification
                reg_loss = regression_criterion(reg_outputs, reg_targets)
                loss = 0.6 * class_loss + 0.4 * reg_loss  # Adjusted loss weights
                loss = loss / accumulation_steps  # Divide loss by accumulation steps

            scaler.scale(loss).backward()

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
        correct_predictions, total_predictions = 0, 0

        with torch.no_grad():
            for inputs, class_targets, reg_targets in val_loader:
                inputs, class_targets, reg_targets = inputs.to(device), class_targets.to(device), reg_targets.to(device)
                class_targets = class_targets.squeeze(1).long()

                with autocast(device_type="cuda", enabled=True):
                    class_outputs, reg_outputs = model(inputs)
                    class_loss = classification_criterion(class_outputs, class_targets)
                    reg_loss = regression_criterion(reg_outputs, reg_targets)
                    val_loss += 0.6 * class_loss.item() + 0.4 * reg_loss.item()

                    # Classification Accuracy
                    predicted_classes = torch.argmax(class_outputs, dim=1)
                    correct_predictions += (predicted_classes == class_targets).sum().item()
                    total_predictions += class_targets.numel()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        classification_accuracy = correct_predictions / total_predictions

        # Logging
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Class Accuracy: {classification_accuracy:.4f}")

        scheduler.step()

    plot_losses(train_losses, val_losses)

def evaluate_model(model, test_loader):
    """
    Evaluates the trained model on the test set.
    """
    model.eval()
    test_loss = 0.0
    correct_predictions, total_predictions = 0, 0

    with torch.no_grad():
        for inputs, class_targets, reg_targets in test_loader:
            inputs, class_targets, reg_targets = inputs.to(device), class_targets.to(device), reg_targets.to(device)
            class_targets = class_targets.squeeze(1).long()

            with autocast(device_type="cuda", enabled=True):
                class_outputs, reg_outputs = model(inputs)
                class_loss = classification_criterion(class_outputs, class_targets)
                reg_loss = regression_criterion(reg_outputs, reg_targets)
                test_loss += 0.6 * class_loss.item() + 0.4 * reg_loss.item()

                # Classification Accuracy
                predicted_classes = torch.argmax(class_outputs, dim=1)
                correct_predictions += (predicted_classes == class_targets).sum().item()
                total_predictions += class_targets.numel()

    test_loss /= len(test_loader)
    classification_accuracy = correct_predictions / total_predictions
    print(f"Test Loss: {test_loss:.4f}, Classification Accuracy: {classification_accuracy:.4f}")
    return test_loss

if __name__ == '__main__':
    # File paths for all forests and their time periods
    forest_csvs = [
        "E:/Sentinelv3/Fazenda Forest/Fazenda_2015_2016.csv",
        "E:/Sentinelv3/Fazenda Forest/Fazenda_2017_2018.csv",
        "E:/Sentinelv3/Fazenda Forest/Fazenda_2019_2020.csv",
        "E:/Sentinelv3/Fazenda Forest/Fazenda_2021_2022.csv",
        "E:/Sentinelv3/Fazenda Forest/Fazenda_2023_2024.csv",
        "E:/Sentinelv3/Rio Aruana Forest/Rio_Aruana_2015_2016.csv",
        "E:/Sentinelv3/Rio Aruana Forest/Rio_Aruana_2017_2018.csv",
        "E:/Sentinelv3/Rio Aruana Forest/Rio_Aruana_2019_2020.csv",
        "E:/Sentinelv3/Rio Aruana Forest/Rio_Aruana_2021_2022.csv",
        "E:/Sentinelv3/Rio Aruana Forest/Rio_Aruana_2023_2024.csv",
        "E:/Sentinelv3/Para Forest/Para_2015_2016.csv",
        "E:/Sentinelv3/Para Forest/Para_2017_2018.csv",
        "E:/Sentinelv3/Para Forest/Para_2019_2020.csv",
        "E:/Sentinelv3/Para Forest/Para_2021_2022.csv",
        "E:/Sentinelv3/Para Forest/Para_2023_2024.csv",
        "E:/Sentinelv3/Braunlage Forest/Braunlage_2015_2016.csv",
        "E:/Sentinelv3/Braunlage Forest/Braunlage_2017_2018.csv",
        "E:/Sentinelv3/Braunlage Forest/Braunlage_2019_2020.csv",
        "E:/Sentinelv3/Braunlage Forest/Braunlage_2021_2022.csv",
        "E:/Sentinelv3/Braunlage Forest/Braunlage_2023_2024.csv",
        "E:/Sentinelv3/Cariboo Forest/Cariboo_2015_2016.csv",
        "E:/Sentinelv3/Cariboo Forest/Cariboo_2017_2018.csv",
        "E:/Sentinelv3/Cariboo Forest/Cariboo_2019_2020.csv",
        "E:/Sentinelv3/Cariboo Forest/Cariboo_2021_2022.csv",
        "E:/Sentinelv3/Cariboo Forest/Cariboo_2023_2024.csv",
        "E:/Sentinelv3/Fort McMurray Forest/Fort_McMurray_2015_2016.csv",
        "E:/Sentinelv3/Fort McMurray Forest/Fort_McMurray_2017_2018.csv",
        "E:/Sentinelv3/Fort McMurray Forest/Fort_McMurray_2019_2020.csv",
        "E:/Sentinelv3/Fort McMurray Forest/Fort_McMurray_2021_2022.csv",
        "E:/Sentinelv3/Fort McMurray Forest/Fort_McMurray_2023_2024.csv",
        "E:/Sentinelv3/Sam Houston Forest/Sam_Houston_2015_2016.csv",
        "E:/Sentinelv3/Sam Houston Forest/Sam_Houston_2017_2018.csv",
        "E:/Sentinelv3/Sam Houston Forest/Sam_Houston_2019_2020.csv",
        "E:/Sentinelv3/Sam Houston Forest/Sam_Houston_2021_2022.csv",
        "E:/Sentinelv3/Sam Houston Forest/Sam_Houston_2023_2024.csv"
    ]
    deforestation_csv = "E:/Sentinelv3/NDVI_Outputs/deforestation_data.csv"

    # Prepare the data
    train_loader, val_loader, test_loader = prepare_data(forest_csvs, deforestation_csv, batch_size=16)

    # Load the model
    model = ResNet50MultiTask().to(device)

    # Define loss functions
    classification_criterion = FocalLoss(alpha=1, gamma=2)  # Focal Loss for multi-class classification
    regression_criterion = nn.MSELoss()  # MSE for regression

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Cyclic learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode="triangular2")

    # Mixed precision training setup
    scaler = GradScaler()

    # Train the model
    train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=20, accumulation_steps=4)

    # Save the trained model
    torch.save(model.state_dict(), "E:/Models/deforestation_model_resnet50_multitask.pth")
    print("Model saved to E:/Models/deforestation_model_resnet50_multitask.pth")

    # Evaluate the model on the test set
    evaluate_model(model, test_loader)

