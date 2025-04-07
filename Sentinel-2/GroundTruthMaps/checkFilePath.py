import os
import glob
import pandas as pd

def check_csv_file_integrity(csv_file):
    """
    Checks that the CSV file exists on disk and is not empty.
    
    Args:
        csv_file (str): Path to the CSV file.
        
    Returns:
        bool: True if the CSV file exists and is non-empty; False otherwise.
    """
    if not os.path.exists(csv_file):
        print(f"[ERROR] CSV file does not exist: {csv_file}")
        return False
    if os.path.getsize(csv_file) == 0:
        print(f"[ERROR] CSV file is empty: {csv_file}")
        return False
    return True

def check_file_paths_in_csv(csv_file, columns_to_check=None):
    """
    Loads a CSV file into a DataFrame and checks that file paths in specified
    columns exist on disk and contain data.
    
    If columns_to_check is None, then every column with the substring "path" 
    (case-insensitive) is checked.
    
    Returns:
        dict: A dictionary mapping each column name to a list of tuples:
              (row index, file_path, error message).
              An empty list for a column means no errors.
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"[ERROR] Could not read {csv_file}: {e}")
        return {}

    # If not specified, check all columns that include "path" (case-insensitive)
    if columns_to_check is None:
        columns_to_check = [col for col in df.columns if "path" in col.lower()]

    missing = {col: [] for col in columns_to_check}

    for idx, row in df.iterrows():
        for col in columns_to_check:
            file_path = str(row[col]).strip()
            if not file_path or file_path.lower() == "nan":
                missing[col].append((idx, file_path, "Empty or NaN value"))
            elif not os.path.exists(file_path):
                missing[col].append((idx, file_path, "File does not exist"))
            else:
                # Check that the file is not empty (has size > 0)
                try:
                    if os.path.getsize(file_path) == 0:
                        missing[col].append((idx, file_path, "File exists but is empty (size = 0)"))
                except Exception as e:
                    missing[col].append((idx, file_path, f"Error checking file size: {e}"))
    return missing

def print_missing_files(csv_file, missing):
    """
    Prints diagnostics for missing or invalid file paths for a CSV file.
    """
    print(f"--- Checking CSV: {csv_file} ---")
    total_issues = 0
    for col, issues in missing.items():
        if issues:
            print(f"In column '{col}':")
            for idx, file_path, message in issues:
                print(f"  Row {idx}: '{file_path}' -> {message}")
            total_issues += len(issues)
        else:
            print(f"Column '{col}' passed: All file paths exist and are non-empty.")
    if total_issues == 0:
        print("All file paths in this CSV are valid.\n")
    else:
        print(f"Total issues found in '{csv_file}': {total_issues}\n")

def main():
    # List of CSVs from your NDVI calculations workflow:
    csv_files = [
        r"E:\Sentinelv3\Combined_Forest_FilePaths.csv",
        r"E:\Sentinelv3\NDVI_Outputs\Processed_Forest_Tiles.csv",
        r"E:\Sentinelv3\NDVI_Outputs\Deforestation_Data_2015_2016_to_2017_2018.csv",
        r"E:\Sentinelv3\NDVI_Outputs\Deforestation_Data_2017_2018_to_2019_2020.csv",
        r"E:\Sentinelv3\NDVI_Outputs\Deforestation_Data_2019_2020_to_2021_2022.csv",
        r"E:\Sentinelv3\NDVI_Outputs\Deforestation_Data_2021_2022_to_2023_2024.csv",
        r"E:\Sentinelv3\NDVI_Outputs\Deforestation_Data_All_Pairs.csv"
    ]
    
    # You can alternatively add CSVs from subfolders automatically using glob:
    # base_output = r"E:\Sentinelv3\NDVI_Outputs"
    # csv_files.extend(glob.glob(os.path.join(base_output, "**", "*.csv"), recursive=True))
    
    for csv_file in csv_files:
        print(f"Checking CSV file: {csv_file}")
        if not check_csv_file_integrity(csv_file):
            continue  # Skip this CSV if it doesn't exist or is empty.
        missing = check_file_paths_in_csv(csv_file)
        print_missing_files(csv_file, missing)

if __name__ == "__main__":
    main()