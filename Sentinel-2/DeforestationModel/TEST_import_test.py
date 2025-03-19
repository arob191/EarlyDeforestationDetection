import torch
import matplotlib.pyplot as plt
from data_preparation import prepare_data

def test_data_preparation():
    # File paths (update with valid file paths if necessary)
    fazenda_csvs = [
        "E:/Sentinelv3/Fazenda Forest/Fazenda_2015_2016.csv",
        "E:/Sentinelv3/Fazenda Forest/Fazenda_2017_2018.csv",
        "E:/Sentinelv3/Fazenda Forest/Fazenda_2019_2020.csv",
        "E:/Sentinelv3/Fazenda Forest/Fazenda_2021_2022.csv",
        "E:/Sentinelv3/Fazenda Forest/Fazenda_2023_2024.csv",
    ]
    deforestation_csv = "E:/Sentinelv3/Fazenda Forest/deforestation_data.csv"

    # Load the DataLoaders
    train_loader, val_loader, test_loader = prepare_data(fazenda_csvs, deforestation_csv, batch_size=1)

    # Fetch a single batch for testing
    for inputs, class_labels, reg_labels in train_loader:
        print(f"Features (inputs) Shape: {inputs.shape}")  # Expected: [1, 2, height, width]
        print(f"Classification Labels Shape: {class_labels.shape}")  # Expected: [1, 1, height, width]
        print(f"Regression Labels Shape: {reg_labels.shape}")  # Expected: [1, 1, height, width] 

        # Plot the inputs and corresponding labels
        plot_sample(inputs[0], class_labels[0], reg_labels[0])
        break  # Test only the first batch

def plot_sample(inputs, class_labels, reg_labels):
    """
    Visualize a single sample from the dataset.
    """
    b4_band = inputs[0].numpy()  # First channel: B4 (Red)
    b8_band = inputs[1].numpy()  # Second channel: B8 (NIR)
    classification_label = class_labels[0].numpy()  # Binary label
    regression_label = reg_labels[0].numpy()  # NDVI difference

    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Plot B4 (Red)
    axes[0].imshow(b4_band, cmap="Reds")
    axes[0].set_title("B4 (Red) Band")
    axes[0].axis("off")

    # Plot B8 (NIR)
    axes[1].imshow(b8_band, cmap="Greens")
    axes[1].set_title("B8 (NIR) Band")
    axes[1].axis("off")

    # Plot Classification Label
    axes[2].imshow(classification_label, cmap="gray")
    axes[2].set_title("Classification Label (Binary)")
    axes[2].axis("off")

    # Plot Regression Label
    im = axes[3].imshow(regression_label, cmap="coolwarm", vmin=-1, vmax=1)
    axes[3].set_title("Regression Label (NDVI Difference)")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# Run the test
if __name__ == "__main__":
    test_data_preparation()