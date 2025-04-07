import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import distance_transform_edt
from multiprocessing import Pool
import csv
import warnings
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# Function to calculate NDVI
def calculate_ndvi_from_multiband(image_path, output_path, target_crs="EPSG:4326"):
    with rasterio.open(image_path) as src:
        red = src.read(1).astype('float32')  # B4 (Red)
        nir = src.read(4).astype('float32')  # B8 (NIR)

        # Calculate NDVI
        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nir - red) / (nir + red)
        ndvi = np.where((nir + red) == 0, 0, ndvi)

        meta = src.meta
        meta.update(dtype='float32', count=1)

        # Reproject to target CRS if necessary
        if src.crs != target_crs:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            meta.update(crs=target_crs, transform=transform, width=width, height=height)

            ndvi_reprojected = np.zeros((height, width), dtype='float32')
            reproject(
                source=ndvi,
                destination=ndvi_reprojected,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )
            ndvi = ndvi_reprojected

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(ndvi, 1)

    return ndvi

# Function to calculate distance maps for spatial context
def calculate_distance_map(binary_map):
    return distance_transform_edt(binary_map == 0)

# Function to process a single tile
def process_tile(row, output_dir, time_period, target_crs="EPSG:4326"):
    tile_id = row['tile_id']
    image_path = row['image_path']
    output_ndvi_path = os.path.join(output_dir, f"{tile_id}_NDVI_{time_period}.tif")
    output_distance_path = os.path.join(output_dir, f"{tile_id}_Distance_{time_period}.tif")

    if os.path.exists(image_path):
        # Calculate NDVI
        ndvi = calculate_ndvi_from_multiband(image_path, output_ndvi_path, target_crs)

        # Threshold for vegetation mask
        threshold = np.percentile(ndvi, 80)  # Example: Top 20% of NDVI values
        binary_map = (ndvi > threshold).astype(np.uint8)
        distance_map = calculate_distance_map(binary_map)

        # Save distance map
        with rasterio.open(output_distance_path, "w", driver='GTiff', height=distance_map.shape[0],
                           width=distance_map.shape[1], count=1, dtype='float32', crs=target_crs) as dst:
            dst.write(distance_map, 1)

        return tile_id, {"ndvi": ndvi, "distance": distance_map}
    else:
        print(f"File not found: {image_path}")
        return tile_id, None

# Function to process all tiles in a time period
def process_tiles(csv_file, output_dir, time_period, target_crs="EPSG:4326"):
    df = pd.read_csv(csv_file)
    processed_results = {}

    with Pool() as pool:
        results = pool.starmap(
            process_tile, [(row, output_dir, time_period, target_crs) for _, row in df.iterrows()]
        )
        for tile_id, layers in results:
            if layers is not None:
                processed_results[tile_id] = layers

    print(f"Processing complete for {time_period}.")
    return processed_results

# Function to calculate NDVI differences and ternary labels
def calculate_ndvi_difference_and_ternary_labels(processed_data1, processed_data2, output_dir, time_period1, time_period2, target_crs="EPSG:4326"):
    deforestation_data = []

    for tile_id in processed_data1:
        if tile_id in processed_data2:
            ndvi_diff = processed_data2[tile_id]["ndvi"] - processed_data1[tile_id]["ndvi"]

            # Dynamic thresholds
            gain_threshold = np.percentile(ndvi_diff.flatten(), 95)
            loss_threshold = np.percentile(ndvi_diff.flatten(), 25)

            # Save NDVI difference map
            diff_output_file = os.path.join(output_dir, f"{tile_id}_NDVI_Diff_{time_period1}_to_{time_period2}.tif")
            with rasterio.open(diff_output_file, "w", driver='GTiff', height=ndvi_diff.shape[0], width=ndvi_diff.shape[1], count=1, dtype='float32', crs=target_crs) as dst:
                dst.write(ndvi_diff, 1)

            # Generate ternary labels
            ternary_labels = np.where(ndvi_diff > gain_threshold, 1,
                             np.where(ndvi_diff < loss_threshold, -1, 0))
            ternary_output_file = os.path.join(output_dir, f"{tile_id}_Ternary_Mask_{time_period1}_to_{time_period2}.tif")
            with rasterio.open(ternary_output_file, "w", driver='GTiff', height=ternary_labels.shape[0],
                               width=ternary_labels.shape[1], count=1, dtype='int16', crs=target_crs) as dst:
                dst.write(ternary_labels, 1)

            # Gather statistics
            total_pixels = ndvi_diff.size
            deforested_pixels = np.sum(ternary_labels == -1)
            vegetation_gain_pixels = np.sum(ternary_labels == 1)
            stable_pixels = np.sum(ternary_labels == 0)
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

# Function to save deforestation data to CSV
def save_deforestation_csv(output_dir, deforestation_data):
    csv_file = os.path.join(output_dir, "deforestation_data.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["tile_id", "time_period", "deforestation_percentage", "ndvi_diff_path", "ternary_mask_path", "vegetation_gain_pixels", "stable_pixels"])
        for entry in deforestation_data:
            writer.writerow([entry["tile_id"], entry["time_period"], entry["deforestation_percentage"], entry["ndvi_diff_path"], entry["ternary_mask_path"], entry["vegetation_gain_pixels"], entry["stable_pixels"]])
    print(f"Deforestation data CSV saved to: {csv_file}")

def reproject_tile(tile_path, target_crs="EPSG:4326"):
    """
    Reprojects a single tile to the target CRS (e.g., EPSG:4326).

    Args:
        tile_path (str): Path to the input tile.
        target_crs (str): Desired CRS for the tile.

    Returns:
        np.ndarray: Reprojected tile array.
        dict: Metadata for the reprojected tile.
    """
    with rasterio.open(tile_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        reprojected_array = np.zeros((height, width), dtype=src.read(1).dtype)
        reproject(
            source=src.read(1),
            destination=reprojected_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )
    return reprojected_array, kwargs

def combine_tiles_with_crs(tile_paths, output_path, grid_size=(10, 10), target_crs="EPSG:4326"):
    """
    Combines tiles into a grid and ensures the result is placed in the original geographic location.

    Args:
        tile_paths (list): List of file paths to individual tiles.
        output_path (str): Path to save the combined tile as a GeoTIFF.
        grid_size (tuple): The number of rows and columns in the grid (e.g., 10x10).
        target_crs (str): Desired CRS for the output image (default is EPSG:4326).

    Returns:
        np.ndarray: Combined grid array.
    """
    if len(tile_paths) != grid_size[0] * grid_size[1]:
        raise ValueError(f"Error: Expected {grid_size[0] * grid_size[1]} tiles, but got {len(tile_paths)}.")

    # Use the metadata from the first tile to detect the original coordinates
    with rasterio.open(tile_paths[0]) as src0:
        first_bounds = src0.bounds  # Geographic bounds of the first tile
        resolution_x = (first_bounds.right - first_bounds.left) / src0.width  # Pixel size (x)
        resolution_y = (first_bounds.top - first_bounds.bottom) / src0.height  # Pixel size (y)
        upper_left_x = first_bounds.left
        upper_left_y = first_bounds.top
        first_crs = src0.crs  # CRS of the first tile

    # Rearrange tiles to ensure proper grid order (bottom-to-top layout)
    reordered_tiles = []
    for row_idx in range(grid_size[0]):
        start_idx = (grid_size[0] - 1 - row_idx) * grid_size[1]  # Reverse rows
        end_idx = start_idx + grid_size[1]
        reordered_tiles.extend(tile_paths[start_idx:end_idx])

    # Initialize a list to store rows
    rows = []

    # Combine rows of tiles
    for row_idx in range(grid_size[0]):
        row_start = row_idx * grid_size[1]
        row_end = row_start + grid_size[1]
        row_tiles = []

        for tile_path in reordered_tiles[row_start:row_end]:
            with rasterio.open(tile_path) as src:
                tile_data = src.read(1)
                row_tiles.append(tile_data)

        # Horizontally stack tiles in the row
        rows.append(np.hstack(row_tiles))

    # Vertically stack all rows to form the full grid
    combined_array = np.vstack(rows)

    # Validate dimensions
    expected_width = grid_size[1] * 100  # 10 tiles per row x 100 pixels each
    expected_height = grid_size[0] * 100  # 10 tiles per column x 100 pixels each
    if combined_array.shape != (expected_height, expected_width):
        print(f"Warning: Combined array dimensions {combined_array.shape} exceed expected {expected_height}x{expected_width}. Clipping to correct size.")
        combined_array = combined_array[:expected_height, :expected_width]  # Clip excess rows/columns

    # Calculate transform for the combined grid
    transform = from_origin(upper_left_x, upper_left_y, resolution_x, resolution_y)

    # Update metadata
    meta = {
        "driver": "GTiff",
        "height": combined_array.shape[0],
        "width": combined_array.shape[1],
        "count": 1,  # Single band
        "dtype": combined_array.dtype,
        "crs": target_crs,
        "transform": transform
    }

    # Save the combined tile
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(combined_array, 1)

    print(f"Combined tiles saved to: {output_path}")
    return combined_array

# Main logic
if __name__ == '__main__':
    base_dir = 'E:\\Sentinelv3'
    output_base_dir = os.path.join(base_dir, 'NDVI_Outputs')
    forest_directories = [
         'Fazenda Forest',
        'Rio Aruana Forest',  'Para Forest',
        'Braunlage Forest', 'Cariboo Forest', 'Fort McMurray Forest',
        'Sam Houston Forest', 'Oblast Forest', 'Tonkino Forest',
        'Iracema Forest'
    ]
    time_periods = ['2015_2016', '2017_2018', '2019_2020', '2021_2022', '2023_2024']

    all_deforestation_data = []  # Aggregate deforestation statistics

    for forest in forest_directories:
        # Standardize forest names for output directory
        forest_base = forest.replace(' ', '_').replace('_Forest', '')
        forest_path = os.path.join(base_dir, forest)
        output_directory = os.path.join(output_base_dir, forest_base)
        os.makedirs(output_directory, exist_ok=True)

        # Reset NDVI results for the current forest
        ndvi_data = {}

        # Process each time period
        for time_period in time_periods:
            csv_file_name = f"{forest_base}_{time_period}.csv"
            csv_path = os.path.join(forest_path, csv_file_name)

            if os.path.exists(csv_path):
                print(f"Processing tiles for {forest_base} during {time_period}...")
                ndvi_data[time_period] = process_tiles(csv_path, output_directory, time_period)
            else:
                print(f"CSV file not found: {csv_path}")

        # Combine adjacent tiles for each time period
        combined_tile_dir = os.path.join(output_directory, "Combined_Tiles")
        os.makedirs(combined_tile_dir, exist_ok=True)

        for time_period, tiles in ndvi_data.items():
            tile_paths = [os.path.join(output_directory, f"{tile_id}_NDVI_{time_period}.tif") for tile_id in tiles.keys()]
            if len(tile_paths) != 100:  # Example check for missing tiles
                print(f"Warning: Missing tiles for {time_period} in {forest_base}.")
                continue
            combined_output_path = os.path.join(combined_tile_dir, f"{forest_base}_combined_{time_period}.tif")
            combine_tiles_with_crs(tile_paths, output_path=combined_output_path, target_crs="EPSG:4326")

        # Calculate NDVI differences across time periods
        for i in range(len(time_periods) - 1):
            period1, period2 = time_periods[i], time_periods[i + 1]
            if period1 in ndvi_data and period2 in ndvi_data:
                print(f"Calculating NDVI differences for {forest_base} from {period1} to {period2}...")
                deforestation_data = calculate_ndvi_difference_and_ternary_labels(
                    ndvi_data[period1], ndvi_data[period2], output_directory, period1, period2
                )
                all_deforestation_data.extend(deforestation_data)

    # Save aggregated deforestation data to CSV
    save_deforestation_csv(output_base_dir, all_deforestation_data)