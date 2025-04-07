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

def apply_augmentation(image_tensor, reg_tensor, ternary_tensor, target_size):
    """
    Custom data augmentation for PyTorch tensors. Applies resizing and horizontal flipping.
    """
    # Resize the image and labels
    image_tensor = resize_tensor(image_tensor, target_size)
    reg_tensor = resize_tensor(reg_tensor.unsqueeze(0), target_size).squeeze(0)

    # Convert ternary_tensor to float for resizing, then back to int after resizing
    ternary_tensor = resize_tensor(ternary_tensor.float().unsqueeze(0), target_size).squeeze(0).round().long()

    # Ensure valid label values after resizing
    ternary_tensor = torch.clamp(ternary_tensor, min=0, max=2)

    # Apply random horizontal flip
    if torch.rand(1).item() < 0.5:
        image_tensor = torch.flip(image_tensor, dims=[2])  # Horizontal flip for image (width axis)
        reg_tensor = torch.flip(reg_tensor, dims=[1])  # Horizontal flip for regression labels
        ternary_tensor = torch.flip(ternary_tensor, dims=[1])  # Horizontal flip for ternary labels

    return image_tensor, reg_tensor, ternary_tensor

def load_images_and_labels(forest_csvs, deforestation_csv, target_size=(128, 128)):
    """
    Loads images and labels for all forests, applies preprocessing, and ensures shapes are correct.

    Returns:
    - features: Tensor of shape [num_samples, 2, height, width]
    - class_labels: Tensor of shape [num_samples, 1, height, width] (ternary classification labels)
    - reg_labels: Tensor of shape [num_samples, 1, height, width] (NDVI regression labels)
    """
    deforestation_df = pd.read_csv(deforestation_csv)
    features = []
    class_labels = []
    reg_labels = []

    for forest_csv in forest_csvs:
        forest_df = pd.read_csv(forest_csv)
        merged_df = pd.merge(forest_df, deforestation_df, on="tile_id", how="inner")

        for _, row in merged_df.iterrows():
            image_path = row["image_path"]
            ndvi_diff_path = row["ndvi_diff_path"]
            ternary_mask_path = row["ternary_mask_path"]

            if os.path.exists(image_path) and os.path.exists(ndvi_diff_path) and os.path.exists(ternary_mask_path):
                # Load the image and extract channels
                with rasterio.open(image_path) as src:
                    image = src.read()  # Shape: [bands, height, width]
                b4, b8 = image[0, :, :], image[3, :, :]  # B4 (Red) and B8 (NIR)
                image_2channel = np.stack([b4, b8], axis=0).astype("float32")
                image_2channel = torch.tensor(image_2channel)

                # Normalize NDVI values (if necessary)
                image_2channel = image_2channel / 10000.0

                # Load the NDVI difference GeoTIFF (regression target)
                with rasterio.open(ndvi_diff_path) as reg_src:
                    ndvi_diff = reg_src.read(1)  # Single-band raster (raw NDVI difference)
                reg_tensor = torch.tensor(ndvi_diff, dtype=torch.float32)

                # Load ternary mask GeoTIFF (classification target)
                with rasterio.open(ternary_mask_path) as ternary_src:
                    ternary_labels = ternary_src.read(1)  # Single-band raster (ternary labels: 1, -1, 0)
                ternary_tensor = torch.tensor(ternary_labels, dtype=torch.int8)

                # Apply resizing and augmentations
                image_2channel, reg_tensor, ternary_tensor = apply_augmentation(
                    image_2channel, reg_tensor, ternary_tensor, target_size
                )

                # Append data
                features.append(image_2channel)
                class_labels.append(ternary_tensor)
                reg_labels.append(reg_tensor)

    # Stack the features and labels into batched tensors
    features = torch.stack(features)  # [num_samples, 2, height, width]
    class_labels = torch.stack(class_labels).unsqueeze(1)  # [num_samples, 1, height, width]
    reg_labels = torch.stack(reg_labels).unsqueeze(1)  # [num_samples, 1, height, width]

    return features, class_labels, reg_labels

def prepare_data(forest_csvs, deforestation_csv, batch_size=16):
    """
    Prepares the DataLoaders for training, validation, and testing.

    Returns:
    - train_loader, val_loader, test_loader
    """
    features, class_labels, reg_labels = load_images_and_labels(forest_csvs, deforestation_csv)

    dataset = TensorDataset(features, class_labels, reg_labels)

    # Split dataset (optionally consider splitting by time periods for testing future predictions)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader