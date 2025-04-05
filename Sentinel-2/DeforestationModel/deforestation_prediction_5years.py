import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model_definition import ResNet34MultiTask  # Ensure this points to your multitask model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """
    Loads the trained ResNet-34 multitask model.
    """
    model = ResNet34MultiTask().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def normalize_input(b4, b8):
    """
    Normalizes the input bands based on training distribution.
    """
    b4 = (b4.astype("float32") / 10000.0)
    b8 = (b8.astype("float32") / 10000.0)
    return np.stack([b4, b8], axis=0)

def predict_deforestation(model, input_tensor, target_size=(128, 128)):
    """
    Predicts regression heat maps for deforestation/growth using the model.
    """
    input_tensor = F.interpolate(input_tensor, size=target_size, mode='bilinear', align_corners=False)

    with torch.no_grad():
        _, reg_predictions = model(input_tensor)
        reg_heat_map = reg_predictions.squeeze().cpu().numpy()
    
    return reg_heat_map

def generate_5_year_prediction(model, image_path, target_size=(128, 128)):
    """
    Simulates deforestation changes over 5 years and outputs regression heat maps for each year.
    """
    # Load the input satellite image
    with rasterio.open(image_path) as src:
        image = src.read()
    b4, b8 = image[0, :, :], image[3, :, :]
    normalized_input = normalize_input(b4, b8)
    input_tensor = torch.tensor(normalized_input).unsqueeze(0).to(device)

    # Amplification factor for regression output
    amplification_factor = 2000.0  # Increase this value to amplify predicted changes

    # Iterate year by year
    heatmaps = []
    current_b8 = b8.copy()  # Start with the original NIR band
    for year in range(1, 6):  # 5 years
        print(f"Predicting deforestation for Year {year}...")
        reg_heat_map = predict_deforestation(model, input_tensor, target_size)

        # Resize regression output to original dimensions
        reg_heat_map_resized = F.interpolate(
            torch.tensor(reg_heat_map).unsqueeze(0).unsqueeze(0),
            size=current_b8.shape,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        # Amplify the regression output
        reg_heat_map_resized *= amplification_factor

        # Debug output to track updates
        print(f"Year {year} Regression Heat Map Range: min={reg_heat_map_resized.min()}, max={reg_heat_map_resized.max()}")

        # Save current year's regression output
        heatmaps.append(reg_heat_map_resized)

        # Update the NIR band with resized and amplified regression output
        current_b8 = current_b8 + reg_heat_map_resized  # Accumulate NDVI changes
        print(f"Year {year} Updated NIR Band Range: min={current_b8.min()}, max={current_b8.max()}")

        # Normalize the updated input
        normalized_next_input = normalize_input(b4, current_b8)
        input_tensor = torch.tensor(normalized_next_input).unsqueeze(0).to(device)

    return heatmaps

def save_heatmap_as_png(heat_map, reference_image_path, year):
    """
    Saves the heat map for a specific year as a PNG file with overlay.
    """
    with rasterio.open(reference_image_path) as src:
        image = src.read(1)  # Use the first band for visualization
        image_height, image_width = image.shape

    # Normalize the heat map for visualization
    normalized_heat_map = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min()) if heat_map.max() > heat_map.min() else heat_map

    # Plot the heat map overlayed on the original image
    plt.figure(figsize=(10, 6))
    plt.imshow(image, cmap="gray", interpolation="none")  # Base satellite image
    plt.imshow(normalized_heat_map, cmap="coolwarm", alpha=0.5, interpolation="none")  # Overlay heat map
    plt.colorbar(label="Deforestation Risk (Year {})".format(year))
    plt.title(f"Year {year} Deforestation Prediction Heat Map")
    plt.axis("off")

    # Save as PNG
    save_path = f"deforestation_heatmap_year_{year}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved heat map to {save_path}")

if __name__ == "__main__":
    image_path = "E:/Sentinelv3/Fazenda Forest/Fazenda_Manna_2015_2016/Fazenda_Manna_2015_2016_Tile_027.tif"
    model_path = "E:/Models/deforestation_model_resnet34_multitask.pth"

    model = load_model(model_path)

    # Generate the regression heat maps for the 5-year period
    yearly_reg_heatmaps = generate_5_year_prediction(model, image_path, target_size=(128, 128))

    # Save heat maps as PNG for each year
    for year, heat_map in enumerate(yearly_reg_heatmaps, start=1):
        save_heatmap_as_png(heat_map, image_path, year)