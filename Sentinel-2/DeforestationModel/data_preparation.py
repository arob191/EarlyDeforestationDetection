import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import pandas as pd
import rasterio
import numpy as np
import os
import torch.nn.functional as F
import warnings
from sklearn.model_selection import train_test_split
from rasterio.errors import NotGeoreferencedWarning
from imblearn.over_sampling import RandomOverSampler
import random

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def resize_tensor(image_tensor, target_size, mode="bilinear"):
    """
    Resizes a tensor using PyTorch interpolation.
    """
    if mode in ["bilinear", "linear", "bicubic", "trilinear"]:
        return F.interpolate(
            image_tensor.unsqueeze(0), size=target_size, mode=mode, align_corners=False
        ).squeeze(0)
    else:
        return F.interpolate(image_tensor.unsqueeze(0), size=target_size, mode=mode).squeeze(0)


def apply_augmentation(image_tensor, reg_tensor, ternary_tensor, distance_tensor, target_size):
    """
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    Applies augmentation to input tensors, including resizing, cropping, flipping, and noise addition.
=======
    Applies augmentation techniques like resizing, random cropping, horizontal flipping, 
    and noise addition to tensors.
>>>>>>> f120c77 (Confusion matrix)
=======
    Applies augmentation techniques like resizing, random cropping, horizontal flipping, 
    and noise addition to tensors.
>>>>>>> f120c77 (Confusion matrix)
=======
    Applies augmentation techniques like resizing, random cropping, horizontal flipping, 
    and noise addition to tensors.
>>>>>>> f120c77 (Confusion matrix)
    """
    # Uniform resizing
    image_tensor = resize_tensor(image_tensor, target_size)
    reg_tensor = resize_tensor(reg_tensor.unsqueeze(0), target_size).squeeze(0)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    ternary_tensor = resize_tensor(ternary_tensor.float().unsqueeze(0), target_size, mode="nearest").round().squeeze(0).long()
=======
    ternary_tensor = resize_tensor(ternary_tensor.float().unsqueeze(0), target_size, mode="nearest").squeeze(0).long()
>>>>>>> f120c77 (Confusion matrix)
=======
    ternary_tensor = resize_tensor(ternary_tensor.float().unsqueeze(0), target_size, mode="nearest").squeeze(0).long()
>>>>>>> f120c77 (Confusion matrix)
=======
    ternary_tensor = resize_tensor(ternary_tensor.float().unsqueeze(0), target_size, mode="nearest").squeeze(0).long()
>>>>>>> f120c77 (Confusion matrix)
    distance_tensor = resize_tensor(distance_tensor.unsqueeze(0), target_size).squeeze(0)

    # Random crop
    crop_size = (target_size[0] - 16, target_size[1] - 16)
    if torch.rand(1).item() < 0.5:
        x = torch.randint(0, image_tensor.shape[1] - crop_size[0] + 1, (1,)).item()
        y = torch.randint(0, image_tensor.shape[2] - crop_size[1] + 1, (1,)).item()
        image_tensor = image_tensor[:, x:x + crop_size[0], y:y + crop_size[1]]
        reg_tensor = reg_tensor[x:x + crop_size[0], y:y + crop_size[1]]
        ternary_tensor = ternary_tensor[x:x + crop_size[0], y:y + crop_size[1]]
        distance_tensor = distance_tensor[x:x + crop_size[0], y:y + crop_size[1]]

    # Random horizontal flip
    if torch.rand(1).item() < 0.5:
        image_tensor = torch.flip(image_tensor, dims=[2])
        reg_tensor = torch.flip(reg_tensor, dims=[1])
        ternary_tensor = torch.flip(ternary_tensor, dims=[1])
        distance_tensor = torch.flip(distance_tensor, dims=[1])

    # Add Gaussian noise to the input image
    noise = torch.randn_like(image_tensor) * 0.01
    image_tensor += noise

    # Final resizing to ensure consistent dimensions
    image_tensor = resize_tensor(image_tensor, target_size)
    reg_tensor = resize_tensor(reg_tensor.unsqueeze(0), target_size).squeeze(0)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    ternary_tensor = resize_tensor(ternary_tensor.float().unsqueeze(0), target_size, mode="nearest").round().squeeze(0).long()
=======
    ternary_tensor = resize_tensor(ternary_tensor.float().unsqueeze(0), target_size, mode="nearest").squeeze(0).long()
>>>>>>> f120c77 (Confusion matrix)
=======
    ternary_tensor = resize_tensor(ternary_tensor.float().unsqueeze(0), target_size, mode="nearest").squeeze(0).long()
>>>>>>> f120c77 (Confusion matrix)
=======
    ternary_tensor = resize_tensor(ternary_tensor.float().unsqueeze(0), target_size, mode="nearest").squeeze(0).long()
>>>>>>> f120c77 (Confusion matrix)
    distance_tensor = resize_tensor(distance_tensor.unsqueeze(0), target_size).squeeze(0)

    return image_tensor, reg_tensor, ternary_tensor, distance_tensor


def check_label_balance(ternary_labels):
    """
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    Checks the distribution of ternary labels across all masks.
=======
    Checks the balance of labels across all ternary masks.
>>>>>>> f120c77 (Confusion matrix)
=======
    Checks the balance of labels across all ternary masks.
>>>>>>> f120c77 (Confusion matrix)
=======
    Checks the balance of labels across all ternary masks.
>>>>>>> f120c77 (Confusion matrix)
    """
    ternary_labels_flat = np.concatenate([label.flatten() for label in ternary_labels])
    unique, counts = np.unique(ternary_labels_flat, return_counts=True)
    total = counts.sum()
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

    # print("Checking label balance across all ternary masks...")
    # print("Proportions of ternary labels:")
    for label, proportion in zip(unique, counts / total * 100):
        class_name = {0: "deforestation", 1: "stable", 2: "gain"}.get(label, "unknown")
        # print(f"{class_name} ({label}): {proportion:.2f}%")

def check_ternary_mask_validity(ternary_tensor):
    """
    Checks if a ternary mask is valid (contains at least two unique classes).
    """
    unique_values = torch.unique(ternary_tensor)
    # print(f"Unique values in ternary mask: {unique_values.tolist()}")  # Debug unique values
=======
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
    proportions = {label: count / total * 100 for label, count in zip(unique, counts)}

    print("Checking label balance across all ternary masks...")
    print("Proportions of ternary labels:")
    for label, proportion in proportions.items():
        class_name = {0: "deforestation", 1: "stable", 2: "gain"}.get(label, "unknown")
        print(f"{class_name} ({label}): {proportion:.2f}%")

def check_ternary_mask_validity(ternary_tensor):
    """
    Validates the ternary mask to ensure it contains diverse classes.
    Returns True if valid (contains at least two unique values), otherwise False.
    """
    unique_values = torch.unique(ternary_tensor)

    # # Debug unique values
    # print(f"Unique values in ternary mask: {unique_values.tolist()}")
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)

    # Valid if at least two unique values are present
    return len(unique_values) > 1

def create_ternary_tensor(ndvi_diff_resized):
    """
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    Creates a ternary tensor based on NDVI differences with thresholds for deforestation and gain.
    """
    ndvi_diff_resized = np.nan_to_num(ndvi_diff_resized, nan=0.0)

    # Adjusted thresholds for balancing classes
    deforestation_threshold = np.percentile(ndvi_diff_resized, 25)
    gain_threshold = np.percentile(ndvi_diff_resized, 85)

    # Debug thresholds and NDVI distribution
=======
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
    Creates a ternary tensor based on NDVI differences using dynamically calculated thresholds.
    Handles cases where NDVI differences contain NaN values.
    """
    # Replace NaN values with zero (or another appropriate value based on your dataset)
    ndvi_diff_resized = np.nan_to_num(ndvi_diff_resized, nan=0.0)

    # Calculate thresholds (25th percentile for deforestation, 95th percentile for gain)
    deforestation_threshold = np.percentile(ndvi_diff_resized, 25)
    gain_threshold = np.percentile(ndvi_diff_resized, 95)

    # # Debug thresholds and NDVI distribution
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
    # print(f"Deforestation threshold: {deforestation_threshold}, Gain threshold: {gain_threshold}")
    # print(f"NDVI differences: Min={ndvi_diff_resized.min()}, Max={ndvi_diff_resized.max()}")

    # Create the ternary tensor
    ternary_tensor = torch.where(
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        torch.tensor(ndvi_diff_resized) < deforestation_threshold, 0,
        torch.where(torch.tensor(ndvi_diff_resized) > gain_threshold, 2, 1)
    )
    # print(f"Unique values in ternary mask: {ternary_tensor.unique().tolist()}")
=======
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
        torch.tensor(ndvi_diff_resized) < deforestation_threshold, 0,  # Class 0: Deforestation
        torch.where(torch.tensor(ndvi_diff_resized) > gain_threshold, 2, 1)  # Class 2: Gain, Class 1: Stable
    )

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
    return ternary_tensor

def load_images_and_labels(forest_csvs, deforestation_csv, target_size=(128, 128)):
    """
    Loads images and labels, applies preprocessing, and ensures consistent tensor sizes.
    Excludes homogeneous ternary masks.
    """
    deforestation_df = pd.read_csv(deforestation_csv)
    features, class_labels, reg_labels = [], [], []

    for forest_csv in forest_csvs:
        forest_df = pd.read_csv(forest_csv)
        merged_df = pd.merge(forest_df, deforestation_df, on="tile_id", how="inner")
        print(f"Preparing {forest_csv} dataset...")

        for _, row in merged_df.iterrows():
            image_path = row["image_path"]
            ndvi_diff_path = row["ndvi_diff_path"]
            distance_map_path = row.get("distance_map_path")

            try:
                # Confirm all required files exist
                if not all(map(os.path.exists, [image_path, ndvi_diff_path])):
                    print(f"Missing or corrupted files: {image_path}, {ndvi_diff_path}")
                    continue

                # Load image
                with rasterio.open(image_path) as src:
                    image = src.read()
                b4, b8 = image[0, :, :], image[3, :, :]  # B4 (Red) and B8 (NIR)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
                # Load and clean NDVI difference
                with rasterio.open(ndvi_diff_path) as ndvi_src:
                    ndvi_diff = ndvi_src.read(1)
                ndvi_diff = np.nan_to_num(ndvi_diff, nan=0.0)  # Replace NaNs
=======
                # Load NDVI difference
                with rasterio.open(ndvi_diff_path) as ndvi_src:
                    ndvi_diff = ndvi_src.read(1)
>>>>>>> f120c77 (Confusion matrix)
=======
                # Load NDVI difference
                with rasterio.open(ndvi_diff_path) as ndvi_src:
                    ndvi_diff = ndvi_src.read(1)
>>>>>>> f120c77 (Confusion matrix)
=======
                # Load NDVI difference
                with rasterio.open(ndvi_diff_path) as ndvi_src:
                    ndvi_diff = ndvi_src.read(1)
>>>>>>> f120c77 (Confusion matrix)

                # Load distance map
                if distance_map_path and os.path.exists(distance_map_path):
                    with rasterio.open(distance_map_path) as dist_src:
                        distance_map = dist_src.read(1)
                    if distance_map.max() > 0:
                        distance_map = distance_map / distance_map.max()  # Normalize
                else:
                    distance_map = np.zeros_like(b4)  # Placeholder if missing

                # Resize arrays
                b4_resized = resize_tensor(torch.tensor(b4).unsqueeze(0), target_size).squeeze(0).numpy()
                b8_resized = resize_tensor(torch.tensor(b8).unsqueeze(0), target_size).squeeze(0).numpy()
                distance_map_resized = resize_tensor(torch.tensor(distance_map).unsqueeze(0), target_size).squeeze(0).numpy()
                ndvi_diff_resized = resize_tensor(torch.tensor(ndvi_diff).unsqueeze(0), target_size).squeeze(0).numpy()

                # Generate the ternary tensor
                ternary_tensor = create_ternary_tensor(ndvi_diff_resized)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

                # Check the validity of the ternary mask
                if not check_ternary_mask_validity(ternary_tensor):
                    print(f"Skipping tile with homogeneous ternary mask: {row['tile_id']}")
                    continue

                # Combine features into a 4-channel tensor
                image_4channel = np.stack([b4_resized, b8_resized, distance_map_resized, ndvi_diff_resized], axis=0).astype("float32") / 10000.0
                image_tensor = torch.tensor(image_4channel)

                # Validate regression tensor
                reg_tensor = torch.tensor(ndvi_diff_resized, dtype=torch.float32)
                if torch.isnan(reg_tensor).any():
                    print(f"Skipping tile {row['tile_id']} due to NaN values in reg_tensor.")
                    continue

=======
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)

                # Check the validity of the ternary mask
                if not check_ternary_mask_validity(ternary_tensor):
                    print(f"Skipping tile with homogeneous ternary mask: {row['tile_id']}")
                    continue

                # Combine features into a 4-channel tensor
                image_4channel = np.stack([b4_resized, b8_resized, distance_map_resized, ndvi_diff_resized], axis=0).astype("float32") / 10000.0
                image_tensor = torch.tensor(image_4channel)
                reg_tensor = torch.tensor(ndvi_diff_resized, dtype=torch.float32)

                # Apply augmentation
                image_tensor, reg_tensor, ternary_tensor, distance_tensor = apply_augmentation(
                    image_tensor, reg_tensor, ternary_tensor, torch.tensor(distance_map_resized), target_size=target_size
                )

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
                # Append processed data
                features.append(image_tensor)
                class_labels.append(ternary_tensor)
                reg_labels.append(reg_tensor)

            except Exception as e:
                print(f"Error processing file: {image_path}. Error: {e}")
                continue

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    # Debug: Check reg_labels before stacking
    # for i, label in enumerate(reg_labels):
    #     print(f"reg_label[{i}] type: {type(label)}, shape: {label.shape if isinstance(label, torch.Tensor) else None}")

=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
    # Stack tensors
    features = torch.stack(features)
    class_labels = torch.stack(class_labels).unsqueeze(1)
    reg_labels = torch.stack(reg_labels).unsqueeze(1)

    # Debug: Check label balance
    check_label_balance(class_labels)

    print(f"Total features collected: {len(features)}")
    return features, class_labels, reg_labels

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
def prepare_data(forest_csvs, deforestation_csv, batch_size=16, target_size=(128, 128)):
    """
    Prepares the DataLoaders for training, validation, and testing with balanced class distributions.
    """
    features, class_labels, reg_labels = load_images_and_labels(forest_csvs, deforestation_csv, target_size=target_size)

    # Assign flat_labels using the majority class in each mask
    flat_labels = torch.tensor([
        random.choices([0, 1, 2], weights=mask.bincount(minlength=3).tolist(), k=1)[0]
        for mask in class_labels.view(len(class_labels), -1)
    ])
    # print(f"Flat labels unique values after revised processing: {flat_labels.unique().tolist()}")

    # Check for sufficient class diversity
    unique_classes = flat_labels.unique()
    # print(f"Classes available before resampling: {unique_classes.tolist()}")
    if len(unique_classes) <= 1:
        raise ValueError(f"Insufficient class diversity. Available classes: {unique_classes.tolist()}")

    # Resample all components to balance classes
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)

    features_flattened = features.view(features.shape[0], -1).numpy()
    flat_labels_numpy = flat_labels.numpy()
    class_labels_flattened = class_labels.view(len(class_labels), -1).numpy()
    reg_labels_flattened = reg_labels.view(len(reg_labels), -1).numpy()

    # print(f"Unique classes in flat_labels_numpy before resampling: {np.unique(flat_labels_numpy)}")
    features_resampled_flattened, flat_labels_resampled = ros.fit_resample(features_flattened, flat_labels_numpy)
    class_labels_resampled, _ = ros.fit_resample(class_labels_flattened, flat_labels_numpy)
    reg_labels_resampled, _ = ros.fit_resample(reg_labels_flattened, flat_labels_numpy)

    # Convert resampled arrays back to tensors
    features_resampled = torch.tensor(features_resampled_flattened).view(-1, features.shape[1], features.shape[2], features.shape[3])
    flat_labels_resampled = torch.tensor(flat_labels_resampled)
    class_labels_resampled = torch.tensor(class_labels_resampled).view(-1, 1, target_size[0], target_size[1])
    reg_labels_resampled = torch.tensor(reg_labels_resampled).view(-1, 1, target_size[0], target_size[1])

    print(f"Unique classes after resampling: {flat_labels_resampled.unique().tolist()}")

    # Perform stratified splitting
    indices = np.arange(len(features_resampled))
    train_indices, temp_indices, _, temp_labels = train_test_split(
        indices, flat_labels_resampled, test_size=0.3, stratify=flat_labels_resampled, random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # Debug splits
    # Display Train, Validation, and Test Class Distributions
    for split_name, split_indices in [("Train", train_indices), ("Validation", val_indices), ("Test", test_indices)]:
        split_labels = flat_labels_resampled[split_indices].numpy()
        unique, counts = np.unique(split_labels, return_counts=True)
        total = sum(counts)
        
        print(f"{split_name} class distribution:")
        for cls, count in zip(unique, counts):
            class_name = {0: "deforestation", 1: "stable", 2: "gain"}.get(cls, "unknown")
            proportion = f"{(count / total) * 100:.2f}%"
            print(f"  {class_name} ({cls}): Count={count}, Proportion={proportion}")

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(features_resampled[train_indices], class_labels_resampled[train_indices], reg_labels_resampled[train_indices]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(features_resampled[val_indices], class_labels_resampled[val_indices], reg_labels_resampled[val_indices]), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(features_resampled[test_indices], class_labels_resampled[test_indices], reg_labels_resampled[test_indices]), batch_size=batch_size, shuffle=False)
=======

def prepare_data(forest_csvs, deforestation_csv, batch_size=16, target_size=(128, 128)):
    """
    Prepares the DataLoaders for training, validation, and testing.
    Excludes tiles with homogeneous ternary masks during preprocessing.

    Args:
        forest_csvs (list): List of file paths to forest-specific CSVs.
        deforestation_csv (str): File path to the deforestation data CSV.
        batch_size (int): Number of samples per batch for DataLoaders.
        target_size (tuple): Target size for resizing images and labels.

    Returns:
        train_loader, val_loader, test_loader (DataLoader): DataLoaders for training, validation, and testing.
    """
    # Load and preprocess features, class labels, and regression labels
    features, class_labels, reg_labels = load_images_and_labels(forest_csvs, deforestation_csv, target_size=target_size)

    # Flatten class labels for stratified split
    flat_labels = class_labels.squeeze(1).reshape(len(features), -1).float().mean(axis=1).round().to(torch.int)

    # Stratified splitting to preserve class distributions
    indices = np.arange(len(features))
    train_indices, temp_indices, _, temp_labels = train_test_split(
        indices, flat_labels, test_size=0.3, stratify=flat_labels, random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # Create TensorDatasets for each split
    train_set = TensorDataset(features[train_indices], class_labels[train_indices], reg_labels[train_indices])
    val_set = TensorDataset(features[val_indices], class_labels[val_indices], reg_labels[val_indices])
    test_set = TensorDataset(features[test_indices], class_labels[test_indices], reg_labels[test_indices])

=======

def prepare_data(forest_csvs, deforestation_csv, batch_size=16, target_size=(128, 128)):
    """
    Prepares the DataLoaders for training, validation, and testing.
    Excludes tiles with homogeneous ternary masks during preprocessing.

    Args:
        forest_csvs (list): List of file paths to forest-specific CSVs.
        deforestation_csv (str): File path to the deforestation data CSV.
        batch_size (int): Number of samples per batch for DataLoaders.
        target_size (tuple): Target size for resizing images and labels.

    Returns:
        train_loader, val_loader, test_loader (DataLoader): DataLoaders for training, validation, and testing.
    """
    # Load and preprocess features, class labels, and regression labels
    features, class_labels, reg_labels = load_images_and_labels(forest_csvs, deforestation_csv, target_size=target_size)

    # Flatten class labels for stratified split
    flat_labels = class_labels.squeeze(1).reshape(len(features), -1).float().mean(axis=1).round().to(torch.int)

    # Stratified splitting to preserve class distributions
    indices = np.arange(len(features))
    train_indices, temp_indices, _, temp_labels = train_test_split(
        indices, flat_labels, test_size=0.3, stratify=flat_labels, random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # Create TensorDatasets for each split
    train_set = TensorDataset(features[train_indices], class_labels[train_indices], reg_labels[train_indices])
    val_set = TensorDataset(features[val_indices], class_labels[val_indices], reg_labels[val_indices])
    test_set = TensorDataset(features[test_indices], class_labels[test_indices], reg_labels[test_indices])

>>>>>>> f120c77 (Confusion matrix)
=======

def prepare_data(forest_csvs, deforestation_csv, batch_size=16, target_size=(128, 128)):
    """
    Prepares the DataLoaders for training, validation, and testing.
    Excludes tiles with homogeneous ternary masks during preprocessing.

    Args:
        forest_csvs (list): List of file paths to forest-specific CSVs.
        deforestation_csv (str): File path to the deforestation data CSV.
        batch_size (int): Number of samples per batch for DataLoaders.
        target_size (tuple): Target size for resizing images and labels.

    Returns:
        train_loader, val_loader, test_loader (DataLoader): DataLoaders for training, validation, and testing.
    """
    # Load and preprocess features, class labels, and regression labels
    features, class_labels, reg_labels = load_images_and_labels(forest_csvs, deforestation_csv, target_size=target_size)

    # Flatten class labels for stratified split
    flat_labels = class_labels.squeeze(1).reshape(len(features), -1).float().mean(axis=1).round().to(torch.int)

    # Stratified splitting to preserve class distributions
    indices = np.arange(len(features))
    train_indices, temp_indices, _, temp_labels = train_test_split(
        indices, flat_labels, test_size=0.3, stratify=flat_labels, random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # Create TensorDatasets for each split
    train_set = TensorDataset(features[train_indices], class_labels[train_indices], reg_labels[train_indices])
    val_set = TensorDataset(features[val_indices], class_labels[val_indices], reg_labels[val_indices])
    test_set = TensorDataset(features[test_indices], class_labels[test_indices], reg_labels[test_indices])

>>>>>>> f120c77 (Confusion matrix)
    # Validate label distributions across splits
    for split_name, split_indices in [("Train", train_indices), ("Validation", val_indices), ("Test", test_indices)]:
        split_labels = flat_labels[split_indices].numpy()
        unique, counts = np.unique(split_labels, return_counts=True)
        print(f"{split_name} label distribution: {dict(zip(unique, counts))}")

    # Create DataLoaders for each dataset
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
>>>>>>> f120c77 (Confusion matrix)

    return train_loader, val_loader, test_loader