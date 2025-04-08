import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Import the dataset and model definitions from your modules.
from data_preparation import DeforestationDataset
from model_definition import DeforestationResNet

def convert_segmentation_mask(pil_img):
    """
    Converts a PIL segmentation mask image into a torch tensor of type Long.
    Assumes the mask contains class indices (e.g., 0, 1, 2) stored as integer pixel values.
    """
    arr = np.array(pil_img, dtype=np.int64)
    return torch.from_numpy(arr)

# Use the custom transform for segmentation target.
target_trans = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.NEAREST),
    convert_segmentation_mask
])

def plot_metrics(train_total_losses, train_seg_losses, train_reg_losses,
                 val_total_losses, val_seg_losses, val_reg_losses, save_path=None):
    """
    Plots training and validation metrics over epochs:
      - Total loss
      - Segmentation loss
      - Regression loss

    If a save_path is provided, the plot is saved to disk.
    """
    epochs = list(range(1, len(train_total_losses) + 1))
    plt.figure(figsize=(12, 8))

    # Plot Total Loss.
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_total_losses, label='Training Total Loss')
    plt.plot(epochs, val_total_losses, label='Validation Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend()

    # Plot Segmentation Loss.
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_seg_losses, label='Training Seg Loss')
    plt.plot(epochs, val_seg_losses, label='Validation Seg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Segmentation Loss')
    plt.legend()

    # Plot Regression Loss.
    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_reg_losses, label='Training Reg Loss')
    plt.plot(epochs, val_reg_losses, label='Validation Reg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Regression Loss')
    plt.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def validate(model, val_loader, device, seg_loss_fn, reg_loss_fn):
    """
    Evaluates the model on the validation set and returns the average losses.
    """
    model.eval()
    total_loss = 0.0
    seg_loss_total = 0.0
    reg_loss_total = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            ndvi_input = batch["input"].to(device)           # [N, 3, 224, 224]
            distance_input = batch["distance"].to(device)      # [N, 1, 224, 224]
            # The segmentation target is returned from our custom transform with shape [H, W]
            # Automatically batched as [N, 224, 224] with type Long.
            seg_target = batch["segmentation"].to(device)      
            reg_target = batch["regression"].to(device).unsqueeze(1)  # [N, 1]

            seg_out, reg_out = model(ndvi_input, distance_input)
            loss_seg = seg_loss_fn(seg_out, seg_target)
            loss_reg = reg_loss_fn(reg_out, reg_target)
            loss = loss_seg + loss_reg

            bs = ndvi_input.size(0)
            total_loss += loss.item() * bs
            seg_loss_total += loss_seg.item() * bs
            reg_loss_total += loss_reg.item() * bs
            count += bs

    model.train()
    return total_loss / count, seg_loss_total / count, reg_loss_total / count

def main():
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters.
    num_epochs = 25
    batch_size = 16
    learning_rate = 1e-4
    num_seg_classes = 3

    # Define input transforms.
    input_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # For the segmentation target, use our custom chain.
    # For the distance map, we use a standard transform.
    distance_trans = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    # Path to the deforestation CSV output by your data preparation script.
    csv_file = r"E:\Sentinelv3\NDVI_Outputs\Deforestation_Data_All_Pairs.csv"
    dataset = DeforestationDataset(csv_file,
                                   transform_input=input_trans,
                                   transform_target=target_trans,
                                   transform_distance=distance_trans)

    # Split the dataset into training (80%) and validation (20%).
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the multi-task model.
    model = DeforestationResNet(num_seg_classes=num_seg_classes, pretrained=True, use_distance=True)
    model = model.to(device)

    # Define loss functions.
    seg_loss_fn = nn.CrossEntropyLoss()  # For segmentation: expects logits [N, C, H, W] and target [N, H, W] of type Long.
    reg_loss_fn = nn.MSELoss()           # For regression: expects scalar outputs [N, 1].
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Containers for storing metrics.
    train_total_losses = []
    train_seg_losses = []
    train_reg_losses = []
    val_total_losses = []
    val_seg_losses = []
    val_reg_losses = []

    # Training loop.
    model.train()
    for epoch in range(num_epochs):
        running_total_loss = 0.0
        running_seg_loss = 0.0
        running_reg_loss = 0.0
        count = 0

        for batch in train_loader:
            ndvi_input = batch["input"].to(device)
            distance_input = batch["distance"].to(device)
            # Our target transform now returns a tensor of shape [H, W] already.
            seg_target = batch["segmentation"].to(device)
            reg_target = batch["regression"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            seg_out, reg_out = model(ndvi_input, distance_input)
            loss_seg = seg_loss_fn(seg_out, seg_target)
            loss_reg = reg_loss_fn(reg_out, reg_target)
            loss = loss_seg + loss_reg

            loss.backward()
            optimizer.step()

            bs = ndvi_input.size(0)
            running_total_loss += loss.item() * bs
            running_seg_loss += loss_seg.item() * bs
            running_reg_loss += loss_reg.item() * bs
            count += bs

        epoch_total_loss = running_total_loss / count
        epoch_seg_loss = running_seg_loss / count
        epoch_reg_loss = running_reg_loss / count

        # Validate the model at the end of the epoch.
        val_total_loss, val_seg_loss, val_reg_loss = validate(model, val_loader, device, seg_loss_fn, reg_loss_fn)

        train_total_losses.append(epoch_total_loss)
        train_seg_losses.append(epoch_seg_loss)
        train_reg_losses.append(epoch_reg_loss)
        val_total_losses.append(val_total_loss)
        val_seg_losses.append(val_seg_loss)
        val_reg_losses.append(val_reg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}]: "
              f"Train Total Loss: {epoch_total_loss:.4f} Seg: {epoch_seg_loss:.4f} Reg: {epoch_reg_loss:.4f} | "
              f"Val Total Loss: {val_total_loss:.4f} Seg: {val_seg_loss:.4f} Reg: {val_reg_loss:.4f}")

    # Plot the training and validation metrics.
    plot_path = os.path.join(r"E:\Sentinelv3\NDVI_Outputs", "training_metrics.png")
    plot_metrics(train_total_losses, train_seg_losses, train_reg_losses,
                 val_total_losses, val_seg_losses, val_reg_losses, save_path=plot_path)

    # Save the trained model checkpoint.
    checkpoint_path = r"E:\Sentinelv3\NDVI_Outputs\deforestation_model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()