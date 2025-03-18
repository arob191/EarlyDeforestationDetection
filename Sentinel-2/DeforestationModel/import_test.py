import matplotlib.pyplot as plt
import torch

def test_data_import(fazenda_csvs, deforestation_csv, target_size=(256, 256)):
    """
    Test script to verify data import and preprocessing.
    """
    from data_preparation import load_images_and_labels  # Ensure `data_preparation.py` is in the same directory

    # Load features and labels
    features, labels = load_images_and_labels(fazenda_csvs, deforestation_csv, target_size=target_size)

    print("Data Import Test:")
    print(f"Features Shape: {features.shape}")  # Should be [num_samples, 2, height, width]
    print(f"Labels Shape: {labels.shape}")      # Should be [num_samples, 1, height, width]

    # Visualize a sample image and label
    sample_idx = 0  # Index of the sample to visualize
    sample_image = features[sample_idx]
    sample_label = labels[sample_idx]

    # Plot the channels of the sample image
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot B4 (Red) band
    axes[0].imshow(sample_image[0].numpy(), cmap='Reds')
    axes[0].set_title("B4 (Red) Band")
    axes[0].axis("off")

    # Plot B8 (NIR) band
    axes[1].imshow(sample_image[1].numpy(), cmap='Greens')
    axes[1].set_title("B8 (NIR) Band")
    axes[1].axis("off")

    # Plot NDVI Difference (Label)
    axes[2].imshow(sample_label[0].numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    axes[2].set_title("NDVI Difference (Label)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

# Example usage
fazenda_csvs = [
    "C:\\Users\\Austin\\OneDrive\\Documents\\Personal Projects\\GitHub\\EarlyDeforestationDetection\\Sentinel-2\\TimePeriods\\Fazenda_2015_2016.csv",
    "C:\\Users\\Austin\\OneDrive\\Documents\\Personal Projects\\GitHub\\EarlyDeforestationDetection\\Sentinel-2\\TimePeriods\\Fazenda_2017_2018.csv"
]
deforestation_csv = "E:\\Sentinelv3\\Fazenda Forest\\NDVI_Outputs\\deforestation_data.csv"

# Run the test
test_data_import(fazenda_csvs, deforestation_csv)