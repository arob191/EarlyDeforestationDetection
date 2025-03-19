import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import pandas as pd
import rasterio
import numpy as np
import os
import torch.nn.functional as F
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

def resize_tensor(image_tensor, target_size):
    """
    Resizes a tensor using PyTorch interpolation.
    """
    return F.interpolate(
        image_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
    ).squeeze(0)

def apply_augmentation(image_tensor, target_size):
    """
    Custom data augmentation for PyTorch tensors.
    """
    # Resize the image
    image_tensor = resize_tensor(image_tensor, target_size)

    # Apply random horizontal flip
    if torch.rand(1).item() < 0.5:
        image_tensor = torch.flip(image_tensor, dims=[2])  # Horizontal flip on width

    return image_tensor

def load_images_and_labels(fazenda_csvs, deforestation_csv, target_size=(128, 128)):  # Reduced input size
    """
    Loads images and labels, applies preprocessing, and ensures shapes are correct.

    Returns:
    - features: Tensor of shape [num_samples, 2, height, width]
    - class_labels: Tensor of shape [num_samples, 1, height, width] (binary classification labels)
    - reg_labels: Tensor of shape [num_samples, 1, height, width] (NDVI regression labels)
    """
    deforestation_df = pd.read_csv(deforestation_csv)
    features = []
    class_labels = []
    reg_labels = []

    for fazenda_csv in fazenda_csvs:
        fazenda_df = pd.read_csv(fazenda_csv)
        merged_df = pd.merge(fazenda_df, deforestation_df, on="tile_id", how="inner")

        for _, row in merged_df.iterrows():
            image_path = row["image_path"]
            label_path = row["ndvi_diff_path"]

            if os.path.exists(image_path) and os.path.exists(label_path):
                # Load the image and extract channels
                with rasterio.open(image_path) as src:
                    image = src.read()  # Shape: [bands, height, width]
                b4, b8 = image[0, :, :], image[3, :, :]
                image_2channel = np.stack([b4, b8], axis=0).astype("float32")
                image_2channel = torch.tensor(image_2channel)

                # Normalize NDVI values (if necessary)
                image_2channel = image_2channel / 10000.0

                # Load the NDVI difference GeoTIFF (regression target)
                with rasterio.open(label_path) as label_src:
                    ndvi_diff = label_src.read(1)  # Single-band raster (raw NDVI difference)
                reg_tensor = torch.tensor(ndvi_diff, dtype=torch.float32)

                # Create classification labels (binary: loss/gain vs. stable vegetation)
                class_tensor = (reg_tensor < -0.2).float()  # 1 for loss, 0 otherwise

                # Resize tensors
                image_2channel = apply_augmentation(image_2channel, target_size)
                reg_tensor = resize_tensor(reg_tensor.unsqueeze(0), target_size).squeeze(0)
                class_tensor = resize_tensor(class_tensor.unsqueeze(0), target_size).squeeze(0)

                # Append data
                features.append(image_2channel)
                class_labels.append(class_tensor)
                reg_labels.append(reg_tensor)

    # Stack the features and labels into batched tensors
    features = torch.stack(features)  # [num_samples, 2, height, width]
    class_labels = torch.stack(class_labels).unsqueeze(1)  # [num_samples, 1, height, width]
    reg_labels = torch.stack(reg_labels).unsqueeze(1)  # [num_samples, 1, height, width]

    return features, class_labels, reg_labels

def prepare_data(fazenda_csvs, deforestation_csv, batch_size=16):  # Smaller batch size
    """
    Prepares the DataLoaders for training, validation, and testing.

    Returns:
    - train_loader, val_loader, test_loader
    """
    features, class_labels, reg_labels = load_images_and_labels(fazenda_csvs, deforestation_csv)

    dataset = TensorDataset(features, class_labels, reg_labels)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


    return train_loader, val_loader, test_loader