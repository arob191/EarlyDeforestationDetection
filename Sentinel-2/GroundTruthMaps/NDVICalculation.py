import os
import pandas as pd
import numpy as np
import rasterio
from multiprocessing import Pool
import csv
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

def calculate_ndvi_from_multiband(image_path, output_path):
    """
    Calculate NDVI from multiband satellite image and save it as a GeoTIFF.
    """
    with rasterio.open(image_path) as src:
        red = src.read(1).astype('float32')  # B4 (Red)
        nir = src.read(4).astype('float32')  # B8 (NIR)

        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nir - red) / (nir + red)
        ndvi = np.where((nir + red) == 0, 0, ndvi)

        meta = src.meta
        meta.update(dtype='float32', count=1)
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(ndvi, 1)

    return ndvi

def process_tile(row, output_dir, time_period):
    """
    Process a single tile to calculate and save NDVI.
    """
    tile_id = row['tile_id']
    image_path = row['image_path']
    output_path = os.path.join(output_dir, f"{tile_id}_NDVI_{time_period}.tif")

    if os.path.exists(image_path):
        return tile_id, calculate_ndvi_from_multiband(image_path, output_path)
    else:
        print(f"File not found: {image_path}")
        return tile_id, None

def process_tiles(csv_file, output_dir, time_period):
    """
    Process all tiles in a given CSV file and calculate NDVI for each.
    """
    df = pd.read_csv(csv_file)
    ndvi_results = {}

    with Pool() as pool:
        results = pool.starmap(
            process_tile, [(row, output_dir, time_period) for _, row in df.iterrows()]
        )
        for tile_id, ndvi in results:
            if ndvi is not None:
                ndvi_results[tile_id] = ndvi

    print(f"NDVI calculation complete for {time_period}.")
    return ndvi_results

def calculate_ndvi_difference_and_ternary_labels(ndvi_dict1, ndvi_dict2, output_dir, time_period1, time_period2, gain_threshold=0.2, loss_threshold=-0.2):
    """
    Calculate NDVI difference maps and generate ternary labels (1, 0, -1) per pixel.
    """
    deforestation_data = []

    for tile_id in ndvi_dict1:
        if tile_id in ndvi_dict2:
            ndvi_diff = ndvi_dict2[tile_id] - ndvi_dict1[tile_id]

            # Save NDVI difference map
            diff_output_file = os.path.join(output_dir, f"{tile_id}_NDVI_Diff_{time_period1}_to_{time_period2}.tif")
            with rasterio.open(diff_output_file, "w", driver='GTiff', height=ndvi_diff.shape[0], width=ndvi_diff.shape[1], count=1, dtype='float32') as dst:
                dst.write(ndvi_diff, 1)

            # Generate ternary labels
            ternary_labels = np.where(ndvi_diff > gain_threshold, 1,     # Vegetation gain
                             np.where(ndvi_diff < loss_threshold, -1, 0))  # Vegetation loss or stable

            # Save ternary labels map
            ternary_output_file = os.path.join(output_dir, f"{tile_id}_Ternary_Mask_{time_period1}_to_{time_period2}.tif")
            with rasterio.open(ternary_output_file, "w", driver='GTiff', height=ternary_labels.shape[0], width=ternary_labels.shape[1], count=1, dtype='int16') as dst:
                dst.write(ternary_labels, 1)

            # Calculate deforestation statistics
            total_pixels = ndvi_diff.size
            deforested_pixels = np.sum(ternary_labels == -1)  # Pixels labeled as vegetation loss
            vegetation_gain_pixels = np.sum(ternary_labels == 1)  # Pixels labeled as vegetation gain
            stable_pixels = np.sum(ternary_labels == 0)  # Pixels labeled as stable
            deforestation_percentage = (deforested_pixels / total_pixels) * 100

            deforestation_data.append({
                "tile_id": tile_id,
                "time_period": f"{time_period1}_to_{time_period2}",
                "deforestation_percentage": deforestation_percentage,
                "ndvi_diff_path": diff_output_file,
                "ternary_mask_path": ternary_output_file,
                "vegetation_gain_pixels": vegetation_gain_pixels,
                "stable_pixels": stable_pixels,
            })

    print(f"NDVI difference and ternary labels calculated for {time_period1} to {time_period2}.")
    return deforestation_data

def save_deforestation_csv(output_dir, deforestation_data):
    """
    Save deforestation data to a CSV file.
    """
    csv_file = os.path.join(output_dir, "deforestation_data.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["tile_id", "time_period", "deforestation_percentage", "ndvi_diff_path", "ternary_mask_path", "vegetation_gain_pixels", "stable_pixels"])
        for entry in deforestation_data:
            writer.writerow([entry["tile_id"], entry["time_period"], entry["deforestation_percentage"], entry["ndvi_diff_path"], entry["ternary_mask_path"], entry["vegetation_gain_pixels"], entry["stable_pixels"]])
    print(f"Deforestation data CSV saved to: {csv_file}")

if __name__ == '__main__':
    base_dir = 'E:\\Sentinelv3'
    output_base_dir = os.path.join(base_dir, 'NDVI_Outputs')
    forest_directories = [
        'Rio Aruana Forest', 'Fazenda Forest', 'Para Forest',
        'Braunlage Forest', 'Cariboo Forest', 'Fort McMurray Forest',
        'Sam Houston Forest'
        ]
    time_periods = ['2015_2016', '2017_2018', '2019_2020', '2021_2022', '2023_2024']

    ndvi_data = {}
    all_deforestation_data = []

    for forest in forest_directories:
        forest_base = forest.replace(' ', '_').replace('_Forest', '')  # Fix for unnecessary suffixes
        forest_path = os.path.join(base_dir, forest)

        for time_period in time_periods:
            # Construct the correct CSV file path for each forest
            csv_file_name = f"{forest_base}_{time_period}.csv"
            csv_path = os.path.join(forest_path, csv_file_name)
            output_directory = os.path.join(output_base_dir, forest_base)

            os.makedirs(output_directory, exist_ok=True)

            if os.path.exists(csv_path):
                ndvi_data[time_period] = process_tiles(csv_path, output_directory, time_period)
            else:
                print(f"CSV not found: {csv_path}")

        for i in range(len(time_periods) - 1):
            period1, period2 = time_periods[i], time_periods[i + 1]
            if period1 in ndvi_data and period2 in ndvi_data:
                deforestation_data = calculate_ndvi_difference_and_ternary_labels(
                    ndvi_data[period1], ndvi_data[period2],
                    output_directory, period1, period2
                )
                all_deforestation_data.extend(deforestation_data)

    save_deforestation_csv(output_base_dir, all_deforestation_data)