import os
import rasterio
import pandas as pd
import numpy as np

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
        # Assuming B4 (Red) is in band 4 and B8 (NIR) is in band 8
        red = src.read(4).astype('float32')  # B4 (Red)
        nir = src.read(8).astype('float32')  # B8 (NIR)

        # Avoid division by zero
        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nir - red) / (nir + red)
        ndvi = np.where((nir + red) == 0, 0, ndvi)  # Set NDVI to 0 where undefined

        # Save NDVI as a new GeoTIFF
        meta = src.meta
        meta.update(dtype='float32', count=1)  # Update metadata for single-band NDVI output
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
        tile_id = row['tile_id']  # Unique tile identifier
        image_path = row['image_path']  # Path to the multi-band image

        if os.path.exists(image_path):
            output_path = os.path.join(output_dir, f"{tile_id}_NDVI_{time_period}.tif")
            ndvi = calculate_ndvi_from_multiband(image_path, output_path)
            ndvi_results[tile_id] = ndvi
        else:
            print(f"File not found: {image_path}")

    print(f"NDVI calculation complete for {time_period}.")
    return ndvi_results

def calculate_ndvi_difference(ndvi_dict1, ndvi_dict2, output_dir, time_period1, time_period2):
    """
    Calculates the NDVI difference between two time periods.

    Parameters:
    ndvi_dict1 (dict): NDVI results for the first time period.
    ndvi_dict2 (dict): NDVI results for the second time period.
    output_dir (str): Directory to save NDVI difference results.
    time_period1 (str): Label for the first time period (e.g., '2015-2016').
    time_period2 (str): Label for the second time period (e.g., '2017-2018').

    Returns:
    None
    """
    for tile_id in ndvi_dict1:
        if tile_id in ndvi_dict2:
            ndvi_diff = ndvi_dict2[tile_id] - ndvi_dict1[tile_id]
            output_file = os.path.join(output_dir, f"{tile_id}_NDVI_Diff_{time_period1}_to_{time_period2}.tif")
            
            with rasterio.open(output_file, "w", driver='GTiff', 
                               height=ndvi_diff.shape[0],
                               width=ndvi_diff.shape[1], 
                               count=1, dtype='float32') as dst:
                dst.write(ndvi_diff, 1)
    print(f"NDVI difference calculated between {time_period1} and {time_period2}.")

# Example usage:
base_csv_dir = "path_to_csv_directory"  # Directory containing your CSV files
output_directory = "path_to_output_directory"  # Directory to save NDVI outputs
time_periods = ["2015-2016", "2017-2018", "2019-2020"]

# Process NDVI for each time period
ndvi_data = {}
for period in time_periods:
    csv_path = os.path.join(base_csv_dir, f"{period}.csv")  # CSV file for the time period
    ndvi_data[period] = process_tiles(csv_path, output_directory, period)

# Calculate NDVI differences between consecutive time periods
calculate_ndvi_difference(ndvi_data["2015-2016"], ndvi_data["2017-2018"], output_directory, "2015-2016", "2017-2018")
calculate_ndvi_difference(ndvi_data["2017-2018"], ndvi_data["2019-2020"], output_directory, "2017-2018", "2019-2020")