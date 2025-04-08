import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

import os
import numpy as np
import pandas as pd
import rasterio

# Hard-coded path to the deforestation data all pairs CSV.
CSV_FILE_PATH = r"E:\Sentinelv3\NDVI_Outputs\Deforestation_Data_All_Pairs.csv"

def is_ndvi_empty(file_path, threshold=1e-6):
    """
    Opens an NDVI difference GeoTIFF, handles NaNs, and returns True
    if the image appears to have no data (i.e. the standard deviation is below the threshold).
    """
    try:
        with rasterio.open(file_path) as src:
            ndvi_diff = src.read(1).astype(np.float32)
    except Exception as e:
        print(f"Error opening '{file_path}': {e}")
        return False  # If we can't open it, we skip it.
    
    # Replace NaN values with zero.
    ndvi_diff = np.nan_to_num(ndvi_diff, nan=0.0)
    
    std = np.std(ndvi_diff)
    # If the standard deviation is essentially zero, treat the image as having no data.
    return std < threshold

def list_empty_ndvi_images(csv_path, ndvi_col="ndvi_diff_path", threshold=1e-6):
    """
    Reads the deforestation data CSV, examines the NDVI difference file paths,
    and prints a list of those images which appear to have no data.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file '{csv_path}': {e}")
        return

    print(f"Loaded CSV with {len(df)} rows.")
    empty_files = []

    for idx, row in df.iterrows():
        file_path = row.get(ndvi_col)
        if not file_path or not isinstance(file_path, str):
            print(f"Row {idx}: No valid file path found in column '{ndvi_col}'. Skipping.")
            continue
        if not os.path.exists(file_path):
            print(f"Row {idx}: File does not exist: {file_path}")
            continue

        if is_ndvi_empty(file_path, threshold=threshold):
            empty_files.append(file_path)

    print("\nThe following NDVI difference file(s) appear to have no data:")
    if empty_files:
        for f in empty_files:
            print(f"  {f}")
    else:
        print("  None. All files appear to contain data.")

def main():
    print(f"Checking NDVI difference images referenced in CSV: {CSV_FILE_PATH}")
    list_empty_ndvi_images(CSV_FILE_PATH, ndvi_col="ndvi_diff_path", threshold=1e-6)

if __name__ == "__main__":
    main()