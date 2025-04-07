import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
import torch.nn.functional as F
from torch.amp import autocast
from data_preparation import prepare_data
from model_definition import ResNet50MultiTask
import matplotlib.pyplot as plt
<<<<<<< HEAD
<<<<<<< HEAD
from sklearn.metrics import mean_absolute_error, f1_score, confusion_matrix, ConfusionMatrixDisplay
=======
from sklearn.metrics import mean_absolute_error, f1_score, confusion_matrix
>>>>>>> f120c77 (Confusion matrix)
=======
from sklearn.metrics import mean_absolute_error, f1_score, confusion_matrix
>>>>>>> f120c77 (Confusion matrix)
import seaborn as sns
import numpy as np
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Focal Loss for multi-class classification
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)  # Prevents log(0)
        F_loss = self.alpha * ((1 - pt + 1e-7) ** self.gamma) * CE_loss  # Added small epsilon for stability
        return F_loss.mean()

def plot_metrics(train_losses, val_losses, classification_accuracies, regression_mae_list):
    """
    Plots training and validation metrics.
    """
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Losses")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(classification_accuracies, label="Classification Accuracy")
    plt.plot(regression_mae_list, label="Regression MAE")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title("Performance Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_metrics.png")  # Save the plot for post-training analysis
    plt.show()

def save_confusion_matrix(y_true, y_pred, labels, file_path, normalize=False):
    """
    Saves a confusion matrix plot with optional normalization.
    """
    y_true_flat = np.concatenate([arr.flatten() for arr in y_true])
    y_pred_flat = np.concatenate([arr.flatten() for arr in y_pred])
    
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize to percentages

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.savefig(file_path)
    plt.close()

def visualize_misclassified_samples(misclassified_samples):
    """
    Visualizes misclassified samples.
    Args:
        misclassified_samples: List of tuples (input_tensor, true_label, predicted_label).
    """
    for i, (input_tensor, true_label, predicted_label) in enumerate(misclassified_samples):
        plt.figure(figsize=(6, 6))
        
        # Handle multi-channel input
        if input_tensor.shape[0] == 1:  # Single-channel (grayscale)
            plt.imshow(input_tensor.squeeze().numpy(), cmap='gray')
        elif input_tensor.shape[0] == 3:  # RGB image
            rgb_image = np.stack([
                input_tensor[0, :, :].numpy(),
                input_tensor[1, :, :].numpy(),
                input_tensor[2, :, :].numpy()
            ], axis=-1)
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())  # Normalize
            plt.imshow(rgb_image)
        else:  # More than 3 channels (e.g., satellite imagery)
            # Visualize the first channel as an example
            plt.imshow(input_tensor[0, :, :].numpy(), cmap='gray')
            plt.title(f"Visualizing Channel 1 of {input_tensor.shape[0]}")

        plt.title(f"True: {true_label}, Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()

<<<<<<< HEAD
<<<<<<< HEAD
def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=20, accumulation_steps=4, early_stop_patience=5):
    """
    Trains a multi-task learning model with gradient accumulation, early stopping, and detailed logging.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        epochs (int): Number of training epochs.
        accumulation_steps (int): Number of steps for gradient accumulation.
        early_stop_patience (int): Number of epochs to wait before early stopping if no improvement.
=======
=======
>>>>>>> f120c77 (Confusion matrix)
def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=2, accumulation_steps=4, early_stop_patience=5):
    """
    Trains the multi-task learning model with early stopping, gradient accumulation,
    and detailed logging for classification and regression metrics.
<<<<<<< HEAD
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
    """
    train_losses = []
    val_losses = []
    classification_accuracies = []
    regression_mae_list = []
<<<<<<< HEAD
<<<<<<< HEAD
    scaler = GradScaler()  # For mixed-precision training
=======
    scaler = GradScaler()
>>>>>>> f120c77 (Confusion matrix)
=======
    scaler = GradScaler()
>>>>>>> f120c77 (Confusion matrix)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc="Training Progress"):
<<<<<<< HEAD
<<<<<<< HEAD
        model.train()
        epoch_train_loss = 0.0
        optimizer.zero_grad()
        opt_step_called = False  # Flag to check if optimizer.step() has been executed in this epoch
=======
=======
>>>>>>> f120c77 (Confusion matrix)
        # Training phase
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
<<<<<<< HEAD
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)

        for i, (inputs, class_targets, reg_targets) in enumerate(train_loader):
            inputs, class_targets, reg_targets = inputs.to(device), class_targets.to(device), reg_targets.to(device)
            class_targets = torch.clamp(class_targets.squeeze(1).long(), min=0, max=2)

<<<<<<< HEAD
<<<<<<< HEAD
            # Validate input data. If NaN or inf values are detected, skip this batch and save the batch for inspection.
            if torch.isnan(inputs).any():
                print(f"[DEBUG] NaN detected in inputs at batch {i}. Saving batch for inspection and skipping this batch.")
                torch.save(inputs, f"debug_inputs_batch_{i}.pt")
                continue
            if torch.isinf(inputs).any():
                print(f"[DEBUG] Inf detected in inputs at batch {i}. Saving batch for inspection and skipping this batch.")
                torch.save(inputs, f"debug_inputs_batch_{i}_inf.pt")
                continue

            # Dynamic loss weighting for classification imbalance
            unique_classes, counts = torch.unique(class_targets, return_counts=True)
            batch_weights = torch.ones(3).to(device)
            for cls, count in zip(unique_classes, counts):
                batch_weights[cls] = 1.0 / (count.item() + 1e-6)

=======
            # Dynamic loss weighting
            unique_classes, counts = torch.unique(class_targets, return_counts=True)
            batch_weights = torch.ones(3).to(device)  # Default weights for classes 0, 1, 2
            for cls, count in zip(unique_classes, counts):
                batch_weights[cls] = 1.0 / (count.item() + 1e-6)

>>>>>>> f120c77 (Confusion matrix)
=======
            # Dynamic loss weighting
            unique_classes, counts = torch.unique(class_targets, return_counts=True)
            batch_weights = torch.ones(3).to(device)  # Default weights for classes 0, 1, 2
            for cls, count in zip(unique_classes, counts):
                batch_weights[cls] = 1.0 / (count.item() + 1e-6)

>>>>>>> f120c77 (Confusion matrix)
            classification_criterion = torch.nn.CrossEntropyLoss(weight=batch_weights)

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                class_outputs, reg_outputs = model(inputs)
                class_loss = classification_criterion(class_outputs, class_targets)
                reg_loss = F.l1_loss(reg_outputs, reg_targets)

<<<<<<< HEAD
<<<<<<< HEAD
                # Dynamic loss weighting for regression — capped to prevent extreme scaling
                dynamic_loss_weight = max(0.1, min(reg_targets.std().item(), 5.0)) if reg_targets.std().item() > 1e-6 else 1.0
                loss = (0.6 * class_loss + 0.4 * dynamic_loss_weight * reg_loss) / accumulation_steps

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

=======
                # Prevent extreme loss weight values
                dynamic_loss_weight = max(0.1, min(reg_targets.std().item(), 10))
                loss = (0.6 * class_loss + 0.4 * dynamic_loss_weight * reg_loss) / accumulation_steps

            scaler.scale(loss).backward()
>>>>>>> f120c77 (Confusion matrix)
=======
                # Prevent extreme loss weight values
                dynamic_loss_weight = max(0.1, min(reg_targets.std().item(), 10))
                loss = (0.6 * class_loss + 0.4 * dynamic_loss_weight * reg_loss) / accumulation_steps

            scaler.scale(loss).backward()
>>>>>>> f120c77 (Confusion matrix)
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                opt_step_called = True

            epoch_train_loss += loss.item()

        # Only call scheduler.step() if at least one optimizer.step() occurred in this epoch.
        if opt_step_called:
            scheduler.step()  # Step the scheduler after optimizer.step()
        else:
            print(f"[DEBUG] No optimizer step was called in epoch {epoch+1}; skipping scheduler.step().")

<<<<<<< HEAD
<<<<<<< HEAD
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_predictions, total_predictions = 0, 0
        regression_mae_sum = 0.0
<<<<<<< HEAD
<<<<<<< HEAD
=======
        all_class_targets, all_class_preds = [], []
        misclassified_samples = []
>>>>>>> f120c77 (Confusion matrix)
=======
        all_class_targets, all_class_preds = [], []
        misclassified_samples = []
>>>>>>> f120c77 (Confusion matrix)

        with torch.no_grad():
            for i, (inputs, class_targets, reg_targets) in enumerate(val_loader):
                inputs, class_targets, reg_targets = inputs.to(device), class_targets.to(device), reg_targets.to(device)
                class_targets = torch.clamp(class_targets.squeeze(1).long(), min=0, max=2)
<<<<<<< HEAD
<<<<<<< HEAD
                
                if torch.isnan(inputs).any():
                    print(f"[DEBUG] NaN detected in validation inputs at batch {i}. Skipping this batch.")
                    continue
                if torch.isnan(class_targets).any():
                    print(f"[DEBUG] NaN detected in class_targets at batch {i}. Skipping this batch.")
                    continue
                if torch.isnan(reg_targets).any():
                    print(f"[DEBUG] NaN detected in reg_targets at batch {i}. Skipping this batch.")
                    continue

                class_outputs, reg_outputs = model(inputs)
                class_loss = classification_criterion(class_outputs, class_targets)
                reg_loss = F.l1_loss(reg_outputs, reg_targets)

                val_loss += 0.6 * class_loss.item() + 0.4 * reg_loss.item()

=======

                with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    class_outputs, reg_outputs = model(inputs)
                    class_loss = classification_criterion(class_outputs, class_targets)
                    reg_loss = F.l1_loss(reg_outputs, reg_targets)

                    dynamic_loss_weight = max(0.1, min(reg_targets.std().item(), 10))
                    val_loss += 0.6 * class_loss.item() + 0.4 * dynamic_loss_weight * reg_loss.item()

                    # Classification Metrics
                    predicted_classes = torch.argmax(class_outputs, dim=1)
                    correct_predictions += (predicted_classes == class_targets).sum().item()
                    total_predictions += class_targets.numel()

                    all_class_targets.extend(class_targets.cpu().numpy())
                    all_class_preds.extend(predicted_classes.cpu().numpy())

                    # Misclassification tracking
                    for batch_idx in range(len(inputs)):
                        pred = predicted_classes[batch_idx].flatten().mode().values.item()
                        true = class_targets[batch_idx].flatten().mode().values.item()
                        if true != pred:
                            misclassified_samples.append((inputs[batch_idx].cpu(), true, pred))

                    regression_mae_sum += F.l1_loss(reg_outputs, reg_targets).item()

        val_loss /= len(val_loader)
        classification_accuracy = correct_predictions / total_predictions
        regression_mae = regression_mae_sum / len(val_loader)
        val_losses.append(val_loss)
        classification_accuracies.append(classification_accuracy)
        regression_mae_list.append(regression_mae)

        # Logging
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Classification Accuracy: {classification_accuracy:.4f}, Regression MAE: {regression_mae:.4f}")

        # Debug Misclassified Samples
        print(f"Epoch {epoch + 1}: Misclassified samples: {len(misclassified_samples)}")
        misclassification_summary = {}
        for true, pred in [(true, pred) for _, true, pred in misclassified_samples]:
            misclassification_summary[(true, pred)] = misclassification_summary.get((true, pred), 0) + 1
        print(f"Misclassification summary: {misclassification_summary}")

        # Save confusion matrix
        save_confusion_matrix(all_class_targets, all_class_preds, labels=[0, 1, 2], file_path=f"confusion_matrix_epoch_{epoch + 1}.png")

        # Early Stopping and Checkpoint Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"E:/Models/deforestation_model_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
                break

        scheduler.step()

    # Plot metrics
    plot_metrics(train_losses, val_losses, classification_accuracies, regression_mae_list)

def evaluate_model(model, test_loader):
    """
    Evaluates the trained model on the test set, logs detailed results, and saves metrics.
    """
    print("Starting evaluation...")
    model.eval()
    test_loss = 0.0
    correct_predictions, total_predictions = 0, 0
    regression_mae_sum = 0.0
    all_class_targets, all_class_preds = [], []
    all_reg_targets, all_reg_preds = [], []

    with torch.no_grad():
        from tqdm import tqdm  # Progress bar for evaluation

        for inputs, class_targets, reg_targets in tqdm(test_loader, desc="Evaluating Batch Progress"):
            inputs, class_targets, reg_targets = inputs.to(device), class_targets.to(device), reg_targets.to(device)
            class_targets = torch.clamp(class_targets.squeeze(1).long(), min=0, max=2)  # Assume preprocessing outputs valid labels

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                class_outputs, reg_outputs = model(inputs)
                class_loss = classification_criterion(class_outputs, class_targets)
                reg_loss = regression_criterion(reg_outputs, reg_targets)
                dynamic_loss_weight = max(0.1, min(10.0, 1.0 / (reg_targets.std().item() + 1e-6)))
                test_loss += 0.6 * class_loss.item() + 0.4 * dynamic_loss_weight * reg_loss.item()

                # Classification Metrics
>>>>>>> f120c77 (Confusion matrix)
                predicted_classes = torch.argmax(class_outputs, dim=1)
                correct_predictions += (predicted_classes == class_targets).sum().item()
                total_predictions += class_targets.numel()
                all_class_targets.extend(class_targets.cpu().numpy().flatten())
                all_class_preds.extend(predicted_classes.cpu().numpy().flatten())

<<<<<<< HEAD
                regression_mae_sum += F.l1_loss(reg_outputs, reg_targets).item()
=======

                with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    class_outputs, reg_outputs = model(inputs)
                    class_loss = classification_criterion(class_outputs, class_targets)
                    reg_loss = F.l1_loss(reg_outputs, reg_targets)

                    dynamic_loss_weight = max(0.1, min(reg_targets.std().item(), 10))
                    val_loss += 0.6 * class_loss.item() + 0.4 * dynamic_loss_weight * reg_loss.item()

                    # Classification Metrics
                    predicted_classes = torch.argmax(class_outputs, dim=1)
                    correct_predictions += (predicted_classes == class_targets).sum().item()
                    total_predictions += class_targets.numel()
>>>>>>> f120c77 (Confusion matrix)

                    all_class_targets.extend(class_targets.cpu().numpy())
                    all_class_preds.extend(predicted_classes.cpu().numpy())

                    # Misclassification tracking
                    for batch_idx in range(len(inputs)):
                        pred = predicted_classes[batch_idx].flatten().mode().values.item()
                        true = class_targets[batch_idx].flatten().mode().values.item()
                        if true != pred:
                            misclassified_samples.append((inputs[batch_idx].cpu(), true, pred))

                    regression_mae_sum += F.l1_loss(reg_outputs, reg_targets).item()

        val_loss /= len(val_loader)
<<<<<<< HEAD
        val_losses.append(val_loss)
        classification_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        classification_accuracies.append(classification_accuracy)
        regression_mae = regression_mae_sum / len(val_loader) if len(val_loader) > 0 else float("inf")
        regression_mae_list.append(regression_mae)

        # Logging for current epoch
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Classification Accuracy: {classification_accuracy:.4f}, Regression MAE: {regression_mae:.4f}")
=======
        classification_accuracy = correct_predictions / total_predictions
        regression_mae = regression_mae_sum / len(val_loader)
        val_losses.append(val_loss)
        classification_accuracies.append(classification_accuracy)
        regression_mae_list.append(regression_mae)

        # Logging
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Classification Accuracy: {classification_accuracy:.4f}, Regression MAE: {regression_mae:.4f}")

        # Debug Misclassified Samples
        print(f"Epoch {epoch + 1}: Misclassified samples: {len(misclassified_samples)}")
        misclassification_summary = {}
        for true, pred in [(true, pred) for _, true, pred in misclassified_samples]:
            misclassification_summary[(true, pred)] = misclassification_summary.get((true, pred), 0) + 1
        print(f"Misclassification summary: {misclassification_summary}")

        # Save confusion matrix
        save_confusion_matrix(all_class_targets, all_class_preds, labels=[0, 1, 2], file_path=f"confusion_matrix_epoch_{epoch + 1}.png")

        # Early Stopping and Checkpoint Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"E:/Models/deforestation_model_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
                break
>>>>>>> f120c77 (Confusion matrix)

        # Check for early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "E:/Models/deforestation_model_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
                break

<<<<<<< HEAD
    # Plot training metrics
=======
    # Plot metrics
>>>>>>> f120c77 (Confusion matrix)
    plot_metrics(train_losses, val_losses, classification_accuracies, regression_mae_list)

def evaluate_model(model, test_loader):
    """
    Evaluates the trained model on the test set, logs detailed results, and saves metrics.
    """
    print("Starting evaluation...")
    model.eval()
    test_loss = 0.0
    correct_predictions, total_predictions = 0, 0
    regression_mae_sum = 0.0
    all_class_targets, all_class_preds = [], []
    all_reg_targets, all_reg_preds = [], []

    with torch.no_grad():
<<<<<<< HEAD
        from tqdm import tqdm

        for inputs, class_targets, reg_targets in tqdm(test_loader, desc="Evaluating Batch Progress"):
            inputs, class_targets, reg_targets = inputs.to(device), class_targets.to(device), reg_targets.to(device)
            class_targets = torch.clamp(class_targets.squeeze(1).long(), min=0, max=2)
=======
        from tqdm import tqdm  # Progress bar for evaluation

        for inputs, class_targets, reg_targets in tqdm(test_loader, desc="Evaluating Batch Progress"):
            inputs, class_targets, reg_targets = inputs.to(device), class_targets.to(device), reg_targets.to(device)
            class_targets = torch.clamp(class_targets.squeeze(1).long(), min=0, max=2)  # Assume preprocessing outputs valid labels
>>>>>>> f120c77 (Confusion matrix)

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                class_outputs, reg_outputs = model(inputs)
                class_loss = classification_criterion(class_outputs, class_targets)
                reg_loss = regression_criterion(reg_outputs, reg_targets)
                dynamic_loss_weight = max(0.1, min(10.0, 1.0 / (reg_targets.std().item() + 1e-6)))
                test_loss += 0.6 * class_loss.item() + 0.4 * dynamic_loss_weight * reg_loss.item()

                # Classification Metrics
                predicted_classes = torch.argmax(class_outputs, dim=1)
                correct_predictions += (predicted_classes == class_targets).sum().item()
                total_predictions += class_targets.numel()
                all_class_targets.extend(class_targets.cpu().numpy().flatten())
                all_class_preds.extend(predicted_classes.cpu().numpy().flatten())

=======
>>>>>>> f120c77 (Confusion matrix)
                # Regression Metrics
                regression_mae = F.l1_loss(reg_outputs, reg_targets).item()
                regression_mae_sum += regression_mae
                all_reg_targets.extend(reg_targets.cpu().numpy().flatten())
                all_reg_preds.extend(reg_outputs.cpu().numpy().flatten())

<<<<<<< HEAD
<<<<<<< HEAD
        # Debug label distributions
        true_dist = np.unique(all_class_targets, return_counts=True)
        pred_dist = np.unique(all_class_preds, return_counts=True)
        print(f"True label proportions: {true_dist}")
        print(f"Predicted label proportions: {pred_dist}")
=======
=======
>>>>>>> f120c77 (Confusion matrix)
        # Debug collected labels
        print(f"Test labels distribution (true): {np.unique(all_class_targets, return_counts=True)}")
        print(f"Test labels distribution (predicted): {np.unique(all_class_preds, return_counts=True)}")

    assert len(all_class_targets) == len(all_class_preds), "Mismatch in true and predicted classification labels!"
    assert len(all_reg_targets) == len(all_reg_preds), "Mismatch in true and predicted regression outputs!"
<<<<<<< HEAD
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)

    # Final metrics
    test_loss /= len(test_loader)
    classification_accuracy = correct_predictions / total_predictions
    regression_mae = regression_mae_sum / len(test_loader)
    f1 = f1_score(all_class_targets, all_class_preds, average="weighted")

    # Log metrics
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Classification Accuracy: {classification_accuracy:.4f}")
    print(f"Classification F1-Score: {f1:.4f}")
    print(f"Regression MAE: {regression_mae:.4f}")

<<<<<<< HEAD
<<<<<<< HEAD
    # Save confusion matrices
    conf_matrix = confusion_matrix(all_class_targets, all_class_preds, labels=[0, 1, 2])
    ConfusionMatrixDisplay(conf_matrix).plot(cmap="viridis")
    plt.savefig("test_confusion_matrix.png")

    normalized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(normalized_conf_matrix).plot(cmap="viridis")
    plt.savefig("test_confusion_matrix_normalized.png")
=======
=======
>>>>>>> f120c77 (Confusion matrix)
    # Save results
    with open("test_results.txt", "w") as f:
        for i, (true_class, pred_class, true_reg, pred_reg) in enumerate(
            zip(all_class_targets, all_class_preds, all_reg_targets, all_reg_preds)
        ):
            f.write(f"Sample {i}: True Class={true_class}, Pred Class={pred_class}, "
                    f"True Regression={true_reg:.4f}, Pred Regression={pred_reg:.4f}\n")

    # Save confusion matrix
    save_confusion_matrix(
        all_class_targets, all_class_preds, labels=[0, 1, 2], file_path="test_confusion_matrix.png"
    )
    save_confusion_matrix(
        all_class_targets, all_class_preds, labels=[0, 1, 2], file_path="test_confusion_matrix_normalized.png", normalize=True
    )

    with open("confusion_matrix_summary.txt", "w") as f:
        f.write(np.array_str(confusion_matrix(all_class_targets, all_class_preds, labels=[0, 1, 2])))
<<<<<<< HEAD
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)

if __name__ == '__main__':
    # File paths for all forests and their time periods
    forest_csvs = [
<<<<<<< HEAD
        # "E:/Sentinelv3/Fazenda Forest/Fazenda_2015_2016.csv",
        # "E:/Sentinelv3/Fazenda Forest/Fazenda_2017_2018.csv",
        # "E:/Sentinelv3/Fazenda Forest/Fazenda_2019_2020.csv",
        # "E:/Sentinelv3/Fazenda Forest/Fazenda_2021_2022.csv",
        # "E:/Sentinelv3/Fazenda Forest/Fazenda_2023_2024.csv",
        # "E:/Sentinelv3/Rio Aruana Forest/Rio_Aruana_2015_2016.csv",
        # "E:/Sentinelv3/Rio Aruana Forest/Rio_Aruana_2017_2018.csv",
        # "E:/Sentinelv3/Rio Aruana Forest/Rio_Aruana_2019_2020.csv",
        # "E:/Sentinelv3/Rio Aruana Forest/Rio_Aruana_2021_2022.csv",
        # "E:/Sentinelv3/Rio Aruana Forest/Rio_Aruana_2023_2024.csv",
        # "E:/Sentinelv3/Para Forest/Para_2015_2016.csv",
        # "E:/Sentinelv3/Para Forest/Para_2017_2018.csv",
        # "E:/Sentinelv3/Para Forest/Para_2019_2020.csv",
        # "E:/Sentinelv3/Para Forest/Para_2021_2022.csv",
        # "E:/Sentinelv3/Para Forest/Para_2023_2024.csv",
        # "E:/Sentinelv3/Braunlage Forest/Braunlage_2015_2016.csv",
        # "E:/Sentinelv3/Braunlage Forest/Braunlage_2017_2018.csv",
        # "E:/Sentinelv3/Braunlage Forest/Braunlage_2019_2020.csv",
        # "E:/Sentinelv3/Braunlage Forest/Braunlage_2021_2022.csv",
        # "E:/Sentinelv3/Braunlage Forest/Braunlage_2023_2024.csv",
        # "E:/Sentinelv3/Cariboo Forest/Cariboo_2015_2016.csv",
        # "E:/Sentinelv3/Cariboo Forest/Cariboo_2017_2018.csv",
        # "E:/Sentinelv3/Cariboo Forest/Cariboo_2019_2020.csv",
        # "E:/Sentinelv3/Cariboo Forest/Cariboo_2021_2022.csv",
        # "E:/Sentinelv3/Cariboo Forest/Cariboo_2023_2024.csv",
        # "E:/Sentinelv3/Fort McMurray Forest/Fort_McMurray_2015_2016.csv",
        # "E:/Sentinelv3/Fort McMurray Forest/Fort_McMurray_2017_2018.csv",
        # "E:/Sentinelv3/Fort McMurray Forest/Fort_McMurray_2019_2020.csv",
        # "E:/Sentinelv3/Fort McMurray Forest/Fort_McMurray_2021_2022.csv",
        # "E:/Sentinelv3/Fort McMurray Forest/Fort_McMurray_2023_2024.csv",
        # "E:/Sentinelv3/Sam Houston Forest/Sam_Houston_2015_2016.csv",
        # "E:/Sentinelv3/Sam Houston Forest/Sam_Houston_2017_2018.csv",
        # "E:/Sentinelv3/Sam Houston Forest/Sam_Houston_2019_2020.csv",
        # "E:/Sentinelv3/Sam Houston Forest/Sam_Houston_2021_2022.csv",
        # "E:/Sentinelv3/Sam Houston Forest/Sam_Houston_2023_2024.csv",
        # E:/Sentinelv3/Iracema Forest/Iracema_2015_2016.csv",
        # "E:/Sentinelv3/Iracema Forest/Iracema_2017_2018.csv",
        # "E:/Sentinelv3/Iracema Forest/Iracema_2019_2020.csv",
        # "E:/Sentinelv3/Iracema Forest/Iracema_2021_2022.csv",
        # "E:/Sentinelv3/Iracema Forest/Iracema_2023_2024.csv",
        # "E:/Sentinelv3/Oblast Forest/Oblast_2015_2016.csv",
        # "E:/Sentinelv3/Oblast Forest/Oblast_2017_2018.csv",
        # "E:/Sentinelv3/Oblast Forest/Oblast_2019_2020.csv",
        # "E:/Sentinelv3/Oblast Forest/Oblast_2021_2022.csv",
        # "E:/Sentinelv3/Oblast Forest/Oblast_2023_2024.csv",
        # "E:/Sentinelv3/Tonkino Forest/Tonkino_2015_2016.csv",
        # "E:/Sentinelv3/Tonkino Forest/Tonkino_2017_2018.csv",
        # "E:/Sentinelv3/Tonkino Forest/Tonkino_2019_2020.csv",
        # "E:/Sentinelv3/Tonkino Forest/Tonkino_2021_2022.csv",
        # "E:/Sentinelv3/Tonkino Forest/Tonkino_2023_2024.csv"
=======
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
        "E:/Sentinelv3/Sam Houston Forest/Sam_Houston_2023_2024.csv",
        "E:/Sentinelv3/Iracema Forest/Iracema_2015_2016",
        "E:/Sentinelv3/Iracema Forest/Iracema_2017_2018",
        "E:/Sentinelv3/Iracema Forest/Iracema_2019_2020",
        "E:/Sentinelv3/Iracema Forest/Iracema_2021_2022",
        "E:/Sentinelv3/Iracema Forest/Iracema_2023_2024",
        "E:/Sentinelv3/Oblast Forest/Oblast_2015_2016",
        "E:/Sentinelv3/Oblast Forest/Oblast_2017_2018",
        "E:/Sentinelv3/Oblast Forest/Oblast_2019_2020",
        "E:/Sentinelv3/Oblast Forest/Oblast_2021_2022",
        "E:/Sentinelv3/Oblast Forest/Oblast_2023_2024",
        "E:/Sentinelv3/Tonkino Forest/Tonkino_2015_2016",
        "E:/Sentinelv3/Tonkino Forest/Tonkino_2017_2018",
        "E:/Sentinelv3/Tonkino Forest/Tonkino_2019_2020",
        "E:/Sentinelv3/Tonkino Forest/Tonkino_2021_2022",
        "E:/Sentinelv3/Tonkino Forest/Tonkino_2023_2024",
<<<<<<< HEAD
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
    ]
    deforestation_csv = "E:/Sentinelv3/NDVI_Outputs/deforestation_data.csv"

    # Prepare the data
    print("Preparing data...")
    train_loader, val_loader, test_loader = prepare_data(forest_csvs, deforestation_csv, batch_size=16)

    # Load the model
    print("Loading model...")
    model = ResNet50MultiTask(in_channels=4).to(device)  # Ensure input channels align with your data preparation script

    # Define loss functions
    classification_criterion = FocalLoss(alpha=1, gamma=2)  # Focal Loss for multi-class classification
    regression_criterion = nn.MSELoss()  # MSE for regression

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Cyclic learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode="triangular2")

    # Train the model
    print("Starting training...")
    train_model(
<<<<<<< HEAD
<<<<<<< HEAD
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs=20,
        accumulation_steps=4,
        early_stop_patience=5
=======
        model, train_loader, val_loader, optimizer, scheduler,
        epochs=20, accumulation_steps=4, early_stop_patience=5
>>>>>>> f120c77 (Confusion matrix)
=======
        model, train_loader, val_loader, optimizer, scheduler,
        epochs=20, accumulation_steps=4, early_stop_patience=5
>>>>>>> f120c77 (Confusion matrix)
    )

    # Save the final trained model
    final_model_path = "E:/Models/deforestation_model_resnet50_multitask_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Evaluate the model on the test set
    print("Evaluating the model on the test set...")
    evaluate_model(model, test_loader)