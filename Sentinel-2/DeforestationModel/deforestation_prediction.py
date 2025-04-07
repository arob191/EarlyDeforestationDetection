import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import torch.nn.functional as F
from model_definition import ResNet50MultiTask  # Ensure this points to your multitask model
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """
    Loads the trained ResNet-50 multitask model.
    """
    model = ResNet50MultiTask().to(device)  # Ternary classification outputs 3 channels
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def normalize_input(b4, b8, distance_map=None):
    """
    Normalizes the input bands based on training distribution and adds distance map as the third channel if available.
    """
    b4 = (b4.astype("float32") / 10000.0)
    b8 = (b8.astype("float32") / 10000.0)
    
    if distance_map is None:
        distance_map = np.zeros_like(b4)  # Use zeroed distance map if not provided

    return np.stack([b4, b8, distance_map], axis=0)

def predict_deforestation(model, image_path, target_size=(128, 128), distance_map_path=None):
    """
    Generates heat maps of deforestation/growth predictions for a given image.
    """
    with rasterio.open(image_path) as src:
        image = src.read()
    b4, b8 = image[0, :, :], image[3, :, :]

    # Load distance map if available
    if distance_map_path and os.path.exists(distance_map_path):
        with rasterio.open(distance_map_path) as dist_src:
            distance_map = dist_src.read(1)
    else:
        distance_map = None

    normalized_input = normalize_input(b4, b8, distance_map)
    image_tensor = torch.tensor(normalized_input).unsqueeze(0).to(device)
    image_tensor = F.interpolate(image_tensor, size=target_size, mode='bilinear', align_corners=False)

    print(f"Image tensor shape: {image_tensor.shape}")  # Debugging

    with torch.no_grad():
        class_predictions, reg_predictions = model(image_tensor)
        print(f"Classification predictions shape: {class_predictions.shape}")
        print(f"Regression predictions shape: {reg_predictions.shape}")

        # Process ternary classification predictions
        class_heat_map = F.softmax(class_predictions, dim=1).squeeze(0).cpu().numpy()
        gain_heat_map, stable_heat_map, loss_heat_map = class_heat_map[2], class_heat_map[1], class_heat_map[0]

        print(f"Heat map ranges - Gain: ({gain_heat_map.min()}, {gain_heat_map.max()}), "
              f"Stable: ({stable_heat_map.min()}, {stable_heat_map.max()}), "
              f"Loss: ({loss_heat_map.min()}, {loss_heat_map.max()})")

        # Process regression predictions
        reg_heat_map = reg_predictions.squeeze().cpu().numpy()

    return gain_heat_map, stable_heat_map, loss_heat_map, reg_heat_map

def plot_heatmap(heat_map, image_path, title, save_path=None, cmap="hot"):
    """
    Plots a heat map overlaid on the original image.
    """
    if len(heat_map.shape) != 2:
        raise ValueError(f"Expected a 2D array for heat_map, but got shape {heat_map.shape}")

    with rasterio.open(image_path) as src:
        image = src.read(1)
        image_height, image_width = image.shape

    heat_map_resized = F.interpolate(
        torch.tensor(heat_map).unsqueeze(0).unsqueeze(0),
        size=(image_height, image_width),
        mode='bilinear',
        align_corners=True
    ).squeeze().numpy()

    normalized_heat_map = (heat_map_resized - heat_map_resized.min()) / (heat_map_resized.max() - heat_map_resized.min()) if heat_map_resized.max() > heat_map_resized.min() else heat_map_resized

    plt.figure(figsize=(10, 6))
    plt.imshow(image, cmap="gray", interpolation="none")
    plt.imshow(normalized_heat_map, cmap=cmap, alpha=0.5, interpolation="none")
    plt.colorbar(label="Deforestation Risk")
    plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved heat map to {save_path}")
    else:
        plt.show()

def save_heatmap_as_tif(heat_map, reference_image_path, save_path):
    """
    Saves a heat map as a GeoTIFF file using the spatial metadata of a reference image.

    Parameters:
    - heat_map (np.ndarray): The heat map to save.
    - reference_image_path (str): Path to the reference image for geospatial metadata.
    - save_path (str): Path to save the GeoTIFF.
    """
    with rasterio.open(reference_image_path) as src:
        meta = src.meta.copy()  # Copy metadata from the reference image
        meta.update({
            "dtype": "float32",  # Data type for the heat map
            "count": 1           # Single-band raster
        })

    # Write the heat map to a GeoTIFF
    with rasterio.open(save_path, "w", **meta) as dst:
        dst.write(heat_map.astype("float32"), 1)  # Write as single-band
    print(f"Heat map saved as GeoTIFF to {save_path}")

if __name__ == "__main__":
    image_path = "E:/Sentinelv3/Fazenda Forest/Fazenda_Manna_2015_2016/Fazenda_Manna_2015_2016_Tile_027.tif"
    distance_map_path = "E:/Sentinelv3/Distance_Maps/Fazenda_Manna_2015_2016_Tile_027_Distance_Map.tif"
    model_path = "E:/Models/deforestation_model_resnet50_multitask.pth"

    model = load_model(model_path)

    # Predict heat maps
    gain_heat_map, stable_heat_map, loss_heat_map, reg_heat_map = predict_deforestation(
        model, image_path, target_size=(128, 128), distance_map_path=distance_map_path
    )

    # Save the regression heat map as a GeoTIFF file
    save_heatmap_as_tif(reg_heat_map, image_path, "predicted_deforestation_change.tif")

    # Plot results
    plot_heatmap(gain_heat_map, image_path, "Vegetation Gain Heat Map", save_path="gain_heat_map.png", cmap="Greens")
    plot_heatmap(stable_heat_map, image_path, "Stable Vegetation Heat Map", save_path="stable_heat_map.png", cmap="Blues")
    plot_heatmap(loss_heat_map, image_path, "Vegetation Loss Heat Map", save_path="loss_heat_map.png", cmap="Reds")
    plot_heatmap(reg_heat_map, image_path, "Deforestation Change Magnitude", save_path="regression_heat_map.png", cmap="coolwarm")