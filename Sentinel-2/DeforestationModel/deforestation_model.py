import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import torch.nn.functional as F
from model_definition import ResNet34MultiTask  # Ensure this points to your multitask model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    """
    Loads the trained ResNet-34 multitask model.

    Parameters:
    - model_path (str): Path to the model weights file.

    Returns:
    - torch.nn.Module: Loaded model.
    """
    model = ResNet34MultiTask(num_classes=1).to(device)
    # Using weights_only=True to address the FutureWarning
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def resize_tensor(image_tensor, target_size):
    """
    Resizes a tensor using PyTorch's interpolation.

    Parameters:
    - image_tensor (torch.Tensor): Input tensor.
    - target_size (tuple): Target (height, width) dimensions.

    Returns:
    - torch.Tensor: Resized tensor.
    """
    return F.interpolate(image_tensor, size=target_size, mode='bilinear', align_corners=False)


def normalize_input(b4, b8):
    """
    Normalizes the input bands based on training distribution.

    Parameters:
    - b4 (np.ndarray): B4 (Red) band.
    - b8 (np.ndarray): B8 (NIR) band.

    Returns:
    - torch.Tensor: Normalized input tensor.
    """
    # Normalize B4 and B8 bands to the 0-1 range
    b4 = (b4.astype("float32") / 10000.0)
    b8 = (b8.astype("float32") / 10000.0)
    return np.stack([b4, b8], axis=0)


def predict_deforestation(model, image_path, target_size=(128, 128)):
    """
    Generates heat maps of deforestation/growth predictions for a given image.

    Parameters:
    - model (torch.nn.Module): The trained multitask model.
    - image_path (str): Path to the input image.
    - target_size (tuple): Target size to resize the image.

    Returns:
    - tuple: Binary heat map (classification), Continuous heat map (regression).
    """
    with rasterio.open(image_path) as src:
        image = src.read()  # Read all bands
    b4, b8 = image[0, :, :], image[3, :, :]  # Extract B4 (Red) and B8 (NIR)
    normalized_input = normalize_input(b4, b8)  # Normalize inputs
    image_tensor = torch.tensor(normalized_input).unsqueeze(0).to(device)  # Add batch dimension
    image_tensor = resize_tensor(image_tensor, target_size)

    print(f"Image tensor shape: {image_tensor.shape}")  # Debug shape

    with torch.no_grad():
        class_predictions, reg_predictions = model(image_tensor)
        print(f"Classification predictions shape: {class_predictions.shape}")
        print(f"Regression predictions shape: {reg_predictions.shape}")

        # Process predictions
        class_heat_map = torch.sigmoid(class_predictions).squeeze().cpu().numpy()  # Binary heat map
        reg_heat_map = reg_predictions.squeeze().cpu().numpy()  # Continuous heat map

        # Debug classification prediction range
        print(f"Min classification prediction: {class_heat_map.min()}, Max classification prediction: {class_heat_map.max()}")

    return class_heat_map, reg_heat_map


def plot_heatmap(heat_map, image_path, title, save_path=None, cmap="hot"):
    """
    Plots a heat map overlaid on the original image.

    Parameters:
    - heat_map (np.ndarray): Heat map array.
    - image_path (str): Path to the original satellite image.
    - title (str): Title of the plot.
    - save_path (str, optional): If provided, saves the plot to this path.
    - cmap (str): Colormap for visualization.

    Returns:
    - None
    """
    if len(heat_map.shape) != 2:
        raise ValueError(f"Expected a 2D array for heat_map, but got shape {heat_map.shape}")

    # Load the original satellite image
    with rasterio.open(image_path) as src:
        image = src.read(1)  # Use the first band for visualization
        image_height, image_width = image.shape

    # Resize heat map to match the satellite image dimensions
    heat_map_resized = F.interpolate(
        torch.tensor(heat_map).unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
        size=(image_height, image_width),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0).numpy()  # Remove extra dimensions

    # Handle normalization for uniform values
    if heat_map_resized.max() > heat_map_resized.min():
        normalized_heat_map = (heat_map_resized - heat_map_resized.min()) / (heat_map_resized.max() - heat_map_resized.min())
    else:
        print("Warning: Heat map values are uniform; skipping normalization.")
        normalized_heat_map = heat_map_resized  # Use raw values as fallback

    # Plot the heat map
    plt.figure(figsize=(10, 6))
    plt.imshow(image, cmap="gray", interpolation="none")  # Satellite image
    plt.imshow(normalized_heat_map, cmap=cmap, alpha=0.5, interpolation="none")  # Heat map overlay
    plt.colorbar(label="Deforestation Risk")
    plt.title(title)
    plt.axis("off")

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved heat map to {save_path}")
    else:
        plt.show()


# Example Usage
if __name__ == "__main__":
    image_path = "E:/Sentinelv3/Fazenda Forest/Fazenda_Manna_2015_2016/Fazenda_Manna_2015_2016_Tile_027.tif"
    model_path = "E:/Models/deforestation_model_resnet34_multitask.pth"

    model = load_model(model_path)

    # Predict 1-year and 5-year deforestation heat maps
    class_heat_map, reg_heat_map = predict_deforestation(model, image_path, target_size=(128, 128))

    # Normalize the classification heat map for better visualization
    normalized_class_heat_map = (class_heat_map - class_heat_map.min()) / (class_heat_map.max() - class_heat_map.min()) if class_heat_map.max() > class_heat_map.min() else class_heat_map

    # Optional: Create a binary thresholded heat map for classification
    threshold = 0.5  # Adjust based on your needs
    binary_heat_map = (class_heat_map > threshold).astype(float)

    # Plot and save results
    plot_heatmap(normalized_class_heat_map, image_path, "1-Year Deforestation Risk Heat Map", save_path="1_year_heat_map.png", cmap="hot")
    plot_heatmap(binary_heat_map, image_path, "1-Year High-Risk Areas (Binary)", save_path="1_year_binary_heat_map.png", cmap="gray")
    plot_heatmap(reg_heat_map, image_path, "5-Year Deforestation Change Magnitude", save_path="5_year_heat_map.png", cmap="coolwarm")

