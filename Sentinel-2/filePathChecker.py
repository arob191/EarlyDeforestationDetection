import pandas as pd
import os

# Path to your CSV files
csv_file_path_early = r'C:\Users\Austin\OneDrive\Documents\Personal Projects\GitHub\EarlyDeforestationDetection\Para_Data_2015_2016.csv'
csv_file_path_late = r'C:\Users\Austin\OneDrive\Documents\Personal Projects\GitHub\EarlyDeforestationDetection\Para_Data_2017_2018.csv'

# Function to check file paths
def check_file_paths(csv_file):
    df = pd.read_csv(csv_file)
    missing_files = []
    for path in df['Path']:
        if not os.path.exists(path):
            missing_files.append(path)
    return missing_files

# Check the early and late period CSV files
missing_files_early = check_file_paths(csv_file_path_early)
missing_files_late = check_file_paths(csv_file_path_late)

if missing_files_early:
    print(f"Missing files in {csv_file_path_early}:")
    for file in missing_files_early:
        print(file)

if missing_files_late:
    print(f"Missing files in {csv_file_path_late}:")
    for file in missing_files_late:
        print(file)

if not missing_files_early and not missing_files_late:
    print("All file paths in the CSV files are correct.")
