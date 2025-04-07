import os
import pandas as pd
import numpy as np
import rasterio
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import distance_transform_edt
=======
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import distance_transform_edt
from multiprocessing import Pool
import csv
>>>>>>> f120c77 (Confusion matrix)
import warnings
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# =============================================================================
# PART A – PER-TILE PROCESSING: NDVI & DISTANCE MAPS
# =============================================================================

def calculate_ndvi_from_multiband(image_path, output_ndvi_path, target_crs="EPSG:4326"):
    """
    Calculates NDVI from a multi-band image and writes the NDVI GeoTIFF.
    Assumption: Band 1 = Red and Band 4 = NIR (adjust indices as needed).
    
    NDVI = (NIR - Red) / (NIR + Red + 1e-6)
    
    Returns the computed NDVI array.
    """
=======
# Function to calculate NDVI
def calculate_ndvi_from_multiband(image_path, output_path, target_crs="EPSG:4326"):
>>>>>>> f120c77 (Confusion matrix)
    with rasterio.open(image_path) as src:
        # Read bands (adjust indices for your sensor configuration)
        red = src.read(1).astype('float32')
        nir = src.read(4).astype('float32')
        
        # Compute NDVI while avoiding division by zero
=======
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
# Function to calculate NDVI
def calculate_ndvi_from_multiband(image_path, output_path, target_crs="EPSG:4326"):
    with rasterio.open(image_path) as src:
        red = src.read(1).astype('float32')  # B4 (Red)
        nir = src.read(4).astype('float32')  # B8 (NIR)

        # Calculate NDVI
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndvi = np.where((nir + red) == 0, 0, ndvi)
        
        # Copy metadata and update for a single band output
        meta = src.meta.copy()
        meta.update(dtype='float32', count=1)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        
        # Reproject if source CRS differs from target_crs
        if src.crs != target_crs:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)
            meta.update(crs=target_crs, transform=transform, width=width, height=height)
            
=======
=======
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)

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
>>>>>>> f120c77 (Confusion matrix)

        # Reproject to target CRS if necessary
        if src.crs != target_crs:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            meta.update(crs=target_crs, transform=transform, width=width, height=height)

>>>>>>> f120c77 (Confusion matrix)
            ndvi_reprojected = np.zeros((height, width), dtype='float32')
            reproject(
                source=ndvi,
                destination=ndvi_reprojected,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
<<<<<<< HEAD
                resampling=Resampling.nearest)
            ndvi = ndvi_reprojected
        
        # Write NDVI GeoTIFF to disk
        with rasterio.open(output_ndvi_path, "w", **meta) as dst:
=======
                resampling=Resampling.nearest
            )
            ndvi = ndvi_reprojected

        with rasterio.open(output_path, "w", **meta) as dst:
>>>>>>> f120c77 (Confusion matrix)
            dst.write(ndvi, 1)
    return ndvi

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
def calculate_distance_map(ndvi, output_distance_path, threshold=0.5):
    """
    Generates a binary vegetation mask using a threshold on NDVI, then computes
    the Euclidean distance transform. Saves result as a GeoTIFF.
    """
    # Generate binary mask (1 for vegetation, 0 for non-vegetation)
    binary_map = (ndvi > threshold).astype(np.uint8)
    distance_map = distance_transform_edt(binary_map == 0)
    
    # Create minimal metadata for output (if georeference is needed, adapt accordingly)
    meta = {
        "driver": "GTiff",
        "height": distance_map.shape[0],
        "width": distance_map.shape[1],
        "count": 1,
        "dtype": "float32"
    }
    with rasterio.open(output_distance_path, "w", **meta) as dst:
        dst.write(distance_map, 1)
    return output_distance_path

def process_tile(row, output_dir, target_crs="EPSG:4326"):
    """
    Processes a single tile based on a CSV row. Calculates NDVI and the distance map.
    Expects the CSV row to contain: 'tile_id', 'image_path', 'forest', 'time_period'.
    Output files are saved in a subfolder for the given forest.
    Returns a dict with tile_id, forest, time_period, and output file paths.
    """
    tile_id = row['tile_id']
    image_path = row['image_path']
    # Clean forest name: remove spaces
    forest = row['forest'].replace(" ", "_")
    time_period = row['time_period']
    
    # Create forest-specific folder within the NDVI_Outputs directory
    forest_folder = os.path.join(output_dir, forest)
    os.makedirs(forest_folder, exist_ok=True)
    
    ndvi_output_path = os.path.join(forest_folder, f"{forest}_{time_period}_Tile_{tile_id}_NDVI.tif")
    distance_output_path = os.path.join(forest_folder, f"{forest}_{time_period}_Tile_{tile_id}_Distance.tif")
    
    if os.path.exists(image_path):
        ndvi_array = calculate_ndvi_from_multiband(image_path, ndvi_output_path, target_crs)
        calculate_distance_map(ndvi_array, distance_output_path, threshold=0.5)
        return {
            "tile_id": tile_id,
            "forest": forest,
            "time_period": time_period,
            "ndvi_path": ndvi_output_path,
            "distance_path": distance_output_path
        }
=======
# Function to calculate distance maps for spatial context
def calculate_distance_map(binary_map):
    return distance_transform_edt(binary_map == 0)

# Function to process a single tile
def process_tile(row, output_dir, time_period, target_crs="EPSG:4326"):
    tile_id = row['tile_id']
    image_path = row['image_path']
=======
# Function to calculate distance maps for spatial context
def calculate_distance_map(binary_map):
    return distance_transform_edt(binary_map == 0)

# Function to process a single tile
def process_tile(row, output_dir, time_period, target_crs="EPSG:4326"):
    tile_id = row['tile_id']
    image_path = row['image_path']
>>>>>>> f120c77 (Confusion matrix)
=======
# Function to calculate distance maps for spatial context
def calculate_distance_map(binary_map):
    return distance_transform_edt(binary_map == 0)

# Function to process a single tile
def process_tile(row, output_dir, time_period, target_crs="EPSG:4326"):
    tile_id = row['tile_id']
    image_path = row['image_path']
>>>>>>> f120c77 (Confusion matrix)
=======
# Function to calculate distance maps for spatial context
def calculate_distance_map(binary_map):
    return distance_transform_edt(binary_map == 0)

# Function to process a single tile
def process_tile(row, output_dir, time_period, target_crs="EPSG:4326"):
    tile_id = row['tile_id']
    image_path = row['image_path']
>>>>>>> f120c77 (Confusion matrix)
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
    else:
        print(f"File not found: {image_path}")
        return None

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
def process_tiles_from_csv(input_csv, output_dir, target_crs="EPSG:4326"):
    """
    Reads the combined forest filepaths CSV and processes each tile to calculate NDVI and
    distance maps. Saves a summary CSV of the processed results.
    """
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        res = process_tile(row, output_dir, target_crs)
        if res is not None:
            results.append(res)
    results_df = pd.DataFrame(results)
    output_csv = os.path.join(output_dir, "Processed_Forest_Tiles.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"Processed tile data saved to: {output_csv}")
    return results_df

# =============================================================================
# PART B – TEMPORAL COMPARISON: NDVI DIFFERENCE & TERNARY MASKS
# =============================================================================

def calculate_ndvi_difference_and_ternary_mask(ndvi_array1, ndvi_array2, diff_output_path, mask_output_path):
    """
    Calculates the pixel-wise difference between two NDVI arrays and creates a ternary mask.
    Uses the 95th percentile as the gain_threshold and the 25th percentile as the loss_threshold.
    
    Ternary mask:
      +1 => Vegetation gain (NDVI increase beyond gain_threshold)
      -1 => Deforestation (NDVI decrease below loss_threshold)
       0 => Stable
    
    Saves the NDVI difference and ternary mask as GeoTIFFs.
    """
    ndvi_diff = ndvi_array2 - ndvi_array1
    gain_threshold = np.percentile(ndvi_diff.flatten(), 95)
    loss_threshold = np.percentile(ndvi_diff.flatten(), 25)
    
    ternary_mask = np.where(ndvi_diff > gain_threshold, 1,
                      np.where(ndvi_diff < loss_threshold, -1, 0))
    
    # Create dummy metadata based on the NDVI array shape (adapt if georeferencing is needed)
    meta = {
        "driver": "GTiff",
        "height": ndvi_diff.shape[0],
        "width": ndvi_diff.shape[1],
        "count": 1,
        "dtype": "float32"
    }
    with rasterio.open(diff_output_path, "w", **meta) as dst:
        dst.write(ndvi_diff, 1)
    
    meta.update(dtype='int16')
    with rasterio.open(mask_output_path, "w", **meta) as dst:
        dst.write(ternary_mask, 1)
    
    return diff_output_path, mask_output_path

def process_tile_pair(group, output_dir, target_crs="EPSG:4326", time_period1="2015_2016", time_period2="2017_2018"):
    """
    Processes a tile pair (same forest, same tile_id) by comparing NDVI outputs from two time periods.
    Computes the NDVI difference and generates a ternary mask.
    Expects the group to contain at least one row for each time period.
    Outputs are saved in the forest subfolder.
    Returns a dict with tile_id, forest, comparison string, and output paths.
    """
    # Ensure we have data for both time periods in the group.
    row1 = group[group['time_period'] == time_period1]
    row2 = group[group['time_period'] == time_period2]
    
    if row1.empty or row2.empty:
        print(f"Tile {group.iloc[0]['tile_id']} missing one of the required time periods.")
        return None
    
    ndvi_path1 = row1.iloc[0]['ndvi_path']
    ndvi_path2 = row2.iloc[0]['ndvi_path']
    
    with rasterio.open(ndvi_path1) as src1:
        ndvi1 = src1.read(1).astype('float32')
    with rasterio.open(ndvi_path2) as src2:
        ndvi2 = src2.read(1).astype('float32')
    
    forest = group.iloc[0]['forest']
    tile_id = group.iloc[0]['tile_id']
    comparison_id = f"{time_period1}_to_{time_period2}"
    
    # Ensure outputs go to the forest-specific folder.
    forest_folder = os.path.join(output_dir, forest)
    os.makedirs(forest_folder, exist_ok=True)
    
    diff_output_path = os.path.join(forest_folder, f"{forest}_{comparison_id}_Tile_{tile_id}_NDVI_Diff.tif")
    mask_output_path = os.path.join(forest_folder, f"{forest}_{comparison_id}_Tile_{tile_id}_Ternary_Mask.tif")
    
    calculate_ndvi_difference_and_ternary_mask(ndvi1, ndvi2, diff_output_path, mask_output_path)
    
    return {
        "tile_id": tile_id,
        "forest": forest,
        "time_period_comparison": comparison_id,
        "ndvi_diff_path": diff_output_path,
        "ternary_mask_path": mask_output_path
    }

def process_tile_pairs(processed_df, output_dir, target_crs="EPSG:4326", time_period1="2015_2016", time_period2="2017_2018"):
    """
    Groups the processed tile DataFrame by forest and tile_id, then processes pairs of NDVI images
    for two specified time periods to generate NDVI differences and ternary masks.
    Saves a summary CSV of the results.
    """
    groups = processed_df.groupby(["forest", "tile_id"])
    pair_results = []
    for name, group in groups:
        result = process_tile_pair(group, output_dir, target_crs, time_period1, time_period2)
        if result is not None:
            pair_results.append(result)
    df_pairs = pd.DataFrame(pair_results)
    output_csv = os.path.join(output_dir, f"Deforestation_Data_{time_period1}_to_{time_period2}.csv")
    df_pairs.to_csv(output_csv, index=False)
    print(f"Deforestation data saved to: {output_csv}")
    return df_pairs

def process_all_time_period_pairs(processed_df, output_dir, target_crs="EPSG:4326", time_periods=None):
    """
    Automates processing of all adjacent time period pairs based on a list of time periods.
    For instance, if time_periods = [ "2015_2016", "2017_2018", "2019_2020", "2021_2022", "2023_2024" ],
    it will compare:
       2015_2016 -> 2017_2018,
       2017_2018 -> 2019_2020,
       2019_2020 -> 2021_2022, and
       2021_2022 -> 2023_2024.
    All pair results are concatenated and saved as one CSV.
    """
    if time_periods is None:
        time_periods = ["2015_2016", "2017_2018", "2019_2020", "2021_2022", "2023_2024"]
    
    all_pairs = []
    for i in range(len(time_periods) - 1):
        tp1 = time_periods[i]
        tp2 = time_periods[i + 1]
        print(f"Processing time period pair: {tp1} to {tp2}")
        df_pairs = process_tile_pairs(processed_df, output_dir, target_crs, time_period1=tp1, time_period2=tp2)
        if df_pairs is not None and not df_pairs.empty:
            all_pairs.append(df_pairs)
    if all_pairs:
        combined_df = pd.concat(all_pairs, ignore_index=True)
        combined_csv = os.path.join(output_dir, "Deforestation_Data_All_Pairs.csv")
        combined_df.to_csv(combined_csv, index=False)
        print(f"All deforestation pair data saved to: {combined_csv}")
    else:
        print("No deforestation pair data produced.")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    # Input CSV with combined forest filepaths.
    # This CSV must contain columns: 'forest', 'time_period', 'tile_id', 'image_path'
    input_csv = r"E:\Sentinelv3\Combined_Forest_FilePaths.csv"
    # Output directory for all processing outputs.
    output_dir = r"E:\Sentinelv3\NDVI_Outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # PART A: Process each tile to generate NDVI and distance maps.
    processed_df = process_tiles_from_csv(input_csv, output_dir, target_crs="EPSG:4326")
    
    # PART B: Process pairs for temporal comparisons.
    # This automatically processes all adjacent pairs as defined by the list.
    time_periods = ["2015_2016", "2017_2018", "2019_2020", "2021_2022", "2023_2024"]
    process_all_time_period_pairs(processed_df, output_dir, target_crs="EPSG:4326", time_periods=time_periods)
    
    # OPTIONAL: Further processing (e.g., mosaicking tiles) can be added here.

if __name__ == "__main__":
    main()
=======
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
=======
>>>>>>> f120c77 (Confusion matrix)
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    save_deforestation_csv(output_base_dir, all_deforestation_data)
>>>>>>> f120c77 (Confusion matrix)
=======
    save_deforestation_csv(output_base_dir, all_deforestation_data)
>>>>>>> f120c77 (Confusion matrix)
=======
    save_deforestation_csv(output_base_dir, all_deforestation_data)
>>>>>>> f120c77 (Confusion matrix)
=======
    save_deforestation_csv(output_base_dir, all_deforestation_data)
>>>>>>> f120c77 (Confusion matrix)
