import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import distance_transform_edt
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

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
    with rasterio.open(image_path) as src:
        # Read bands (adjust indices for your sensor configuration)
        red = src.read(1).astype('float32')
        nir = src.read(4).astype('float32')
        
        # Compute NDVI while avoiding division by zero
        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndvi = np.where((nir + red) == 0, 0, ndvi)
        
        # Copy metadata and update for a single band output
        meta = src.meta.copy()
        meta.update(dtype='float32', count=1)
        
        # Reproject if source CRS differs from target_crs
        if src.crs != target_crs:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)
            meta.update(crs=target_crs, transform=transform, width=width, height=height)
            
            ndvi_reprojected = np.zeros((height, width), dtype='float32')
            reproject(
                source=ndvi,
                destination=ndvi_reprojected,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest)
            ndvi = ndvi_reprojected
        
        # Write NDVI GeoTIFF to disk
        with rasterio.open(output_ndvi_path, "w", **meta) as dst:
            dst.write(ndvi, 1)
    return ndvi

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
    else:
        print(f"File not found: {image_path}")
        return None

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