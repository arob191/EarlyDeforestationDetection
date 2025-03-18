import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import pandas as pd
import rasterio
import numpy as np
import os
import torch.nn.functional as F

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
    # Apply random rotation
    angle = torch.randint(-15, 15, (1,)).item()
    grid = F.affine_grid(
        torch.eye(2, 3).unsqueeze(0).to(image_tensor.device),  # Identity affine transform
        image_tensor.unsqueeze(0).size(), align_corners=True  # Explicitly align corners
    )
    image_tensor = F.grid_sample(image_tensor.unsqueeze(0), grid, align_corners=True).squeeze(0)
    return image_tensor

def load_images_and_labels(fazenda_csvs, deforestation_csv, target_size=(256, 256)):
    """
    Loads images and labels, applies preprocessing and augmentation, and ensures shapes are correct.

    Returns:
    - features: Tensor of shape [num_samples, 2, height, width]
    - labels: Tensor of shape [num_samples]
    """
    deforestation_df = pd.read_csv(deforestation_csv)
    features = []
    labels = []

    for fazenda_csv in fazenda_csvs:
        fazenda_df = pd.read_csv(fazenda_csv)
        merged_df = pd.merge(fazenda_df, deforestation_df, on="tile_id")

        for _, row in merged_df.iterrows():
            image_path = row["image_path"]
            label = row["deforestation_percentage"]

            if os.path.exists(image_path):
                # Load the image and extract channels
                with rasterio.open(image_path) as src:
                    image = src.read()  # Shape: [bands, height, width]
                b4, b8 = image[0, :, :], image[3, :, :]
                image_2channel = np.stack([b4, b8], axis=0).astype("float32") / 10000.0
                image_2channel = torch.tensor(image_2channel)

                # Apply augmentation and resizing
                image_2channel = apply_augmentation(image_2channel, target_size)

                # Append the features and scalar labels
                features.append(image_2channel)
                labels.append(torch.tensor(label, dtype=torch.float32))

    # Stack the features and labels into batched tensors
    features = torch.stack(features)  # Shape: [num_samples, 2, height, width]
    labels = torch.stack(labels)      # Shape: [num_samples]
    return features, labels

def prepare_data(fazenda_csvs, deforestation_csv, batch_size=32):
    """
    Prepares the DataLoaders for training, validation, and testing.
    """
    features, labels = load_images_and_labels(fazenda_csvs, deforestation_csv)
    dataset = TensorDataset(features, labels)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader