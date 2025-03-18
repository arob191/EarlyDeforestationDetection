import os
import rasterio
import pandas as pd
import numpy as np
import csv

def calculate_ndvi_from_multiband(image_path, output_path):
    """
    Calculates NDVI from a multi-band GeoTIFF and saves the result.

    Parameters:
    image_path (str): Path to the multi-band GeoTIFF.
    output_path (str): Path to save the NDVI GeoTIFF.

    Returns:
    numpy.ndarray: NDVI array.
    """
    with rasterio.open(image_path) as src:
        # Extract B4 (Red) and B8 (NIR) bands
        red = src.read(1).astype('float32')  # B4 (Red)
        nir = src.read(4).astype('float32')  # B8 (NIR)

        # Avoid division by zero
        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nir - red) / (nir + red)
        ndvi = np.where((nir + red) == 0, 0, ndvi)  # Set NDVI to 0 where undefined

        # Save NDVI as a GeoTIFF
        meta = src.meta
        meta.update(dtype='float32', count=1)
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(ndvi, 1)

    print(f"NDVI saved to: {output_path}")
    return ndvi

def process_tiles(csv_file, output_dir, time_period):
    """
    Processes each tile to calculate NDVI and save results.

    Parameters:
    csv_file (str): Path to CSV file containing image paths for the tiles.
    output_dir (str): Directory to save the NDVI outputs.
    time_period (str): Label for the time period (e.g., '2015-2016').

    Returns:
    dict: Dictionary with NDVI results for all tiles.
    """
    df = pd.read_csv(csv_file)
    ndvi_results = {}

    for idx, row in df.iterrows():
        tile_id = row['tile_id']
        image_path = row['image_path']

        if os.path.exists(image_path):
            output_path = os.path.join(output_dir, f"{tile_id}_NDVI_{time_period}.tif")
            ndvi = calculate_ndvi_from_multiband(image_path, output_path)
            ndvi_results[tile_id] = ndvi
        else:
            print(f"File not found: {image_path}")

    print(f"NDVI calculation complete for {time_period}.")
    return ndvi_results

def calculate_ndvi_difference_and_deforestation(ndvi_dict1, ndvi_dict2, output_dir, time_period1, time_period2):
    """
    Calculates the NDVI difference and deforestation amount for each tile.

    Parameters:
    ndvi_dict1 (dict): NDVI results for the first time period.
    ndvi_dict2 (dict): NDVI results for the second time period.
    output_dir (str): Directory to save NDVI difference results and CSV.
    time_period1 (str): Label for the first time period.
    time_period2 (str): Label for the second time period.

    Returns:
    list: List of dictionaries containing deforestation data for each tile.
    """
    deforestation_data = []

    for tile_id in ndvi_dict1:
        if tile_id in ndvi_dict2:
            ndvi_diff = ndvi_dict2[tile_id] - ndvi_dict1[tile_id]
            output_file = os.path.join(output_dir, f"{tile_id}_NDVI_Diff_{time_period1}_to_{time_period2}.tif")
            
            # Save NDVI difference as GeoTIFF
            with rasterio.open(output_file, "w", driver='GTiff',
                               height=ndvi_diff.shape[0],
                               width=ndvi_diff.shape[1],
                               count=1, dtype='float32') as dst:
                dst.write(ndvi_diff, 1)
            
            # Calculate deforestation percentage
            total_pixels = ndvi_diff.size
            deforested_pixels = np.sum(ndvi_diff < -0.2)  # Threshold for deforestation (adjust as needed)
            deforestation_percentage = (deforested_pixels / total_pixels) * 100

            # Append data for this tile
            deforestation_data.append({
                "tile_id": tile_id,
                "time_period": f"{time_period1}_to_{time_period2}",
                "deforestation_percentage": deforestation_percentage,
                "ndvi_diff_path": output_file
            })

    print(f"NDVI difference calculated and deforestation data collected for {time_period1} to {time_period2}.")
    return deforestation_data

def save_deforestation_csv(output_dir, deforestation_data):
    """
    Save deforestation data into a CSV file.

    Parameters:
    output_dir (str): Directory to save the CSV file.
    deforestation_data (list): List of dictionaries containing deforestation data.

    Returns:
    None
    """
    csv_file = os.path.join(output_dir, "deforestation_data.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["tile_id", "time_period", "deforestation_percentage", "ndvi_diff_path"])
        for entry in deforestation_data:
            writer.writerow([entry["tile_id"], entry["time_period"], entry["deforestation_percentage"], entry["ndvi_diff_path"]])
    print(f"Deforestation data CSV saved to: {csv_file}")

base_csv_dir = 'E:\\Sentinelv3\\Fazenda Forest'
output_directory = 'E:\\Sentinelv3\\Fazenda Forest\\NDVI_Outputs'
forest = "Fazenda"
time_periods = ["2015_2016", "2017_2018", "2019_2020"]

ndvi_data = {}
for period in time_periods:
    csv_path = os.path.join(base_csv_dir, f"{forest}_{period}.csv")
    ndvi_data[period] = process_tiles(csv_path, output_directory, period)

all_deforestation_data = []
all_deforestation_data.extend(calculate_ndvi_difference_and_deforestation(
    ndvi_data["2015_2016"], ndvi_data["2017_2018"], output_directory, "2015_2016", "2017_2018"))
all_deforestation_data.extend(calculate_ndvi_difference_and_deforestation(
    ndvi_data["2017_2018"], ndvi_data["2019_2020"], output_directory, "2017_2018", "2019_2020"))

save_deforestation_csv(output_directory, all_deforestation_data)
