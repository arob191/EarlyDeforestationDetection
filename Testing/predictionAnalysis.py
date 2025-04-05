import numpy as np
import rasterio
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

def load_heatmap(file_path):
    """
    Loads a heat map (GeoTIFF) as a NumPy array.
    """
    with rasterio.open(file_path) as src:
        heatmap = src.read(1)  # Read the first band
    return heatmap

def calculate_metrics(ground_truth, predictions):
    """
    Calculates evaluation metrics for the heat maps.
    """
    mae = mean_absolute_error(ground_truth, predictions)
    mse = mean_squared_error(ground_truth, predictions)
    correlation, _ = pearsonr(ground_truth.flatten(), predictions.flatten())  # Flatten for correlation
    return mae, mse, correlation

def plot_difference_map(ground_truth, predictions, save_path=None):
    """
    Plots the difference between the ground truth and prediction heat maps.
    """
    difference = ground_truth - predictions
    plt.figure(figsize=(10, 6))
    plt.imshow(difference, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    plt.colorbar(label="Difference (Ground Truth - Prediction)")
    plt.title("Difference Map")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved difference map to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # File paths for the ground truth and predicted heat maps
    ground_truth_path = "E:\\Sentinelv3\\NDVI_Outputs\\Fazenda\\27_NDVI_Diff_2015_2016_to_2017_2018.tif"
    prediction_path = prediction_path = "C:\\Users\\Austin\\OneDrive\\Documents\\Personal Projects\\GitHub\\EarlyDeforestationDetection\\predicted_deforestation_change.tif"

    # Load the heat maps
    ground_truth = load_heatmap(ground_truth_path)
    predictions = load_heatmap(prediction_path)

    # Ensure the maps have the same shape
    if ground_truth.shape != predictions.shape:
        raise ValueError("Ground truth and predictions must have the same dimensions!")

    # Calculate metrics
    mae, mse, correlation = calculate_metrics(ground_truth, predictions)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Pearson Correlation: {correlation:.4f}")

    # Plot the difference map
    plot_difference_map(ground_truth, predictions, save_path="difference_map.png")