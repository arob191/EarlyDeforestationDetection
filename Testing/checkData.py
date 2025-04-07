import pandas as pd
import os

def test_file_paths(csv_file, check_file_existence=True):
    """
    Load the CSV file and verify that the image_path column has no null or empty values.
    Optionally, check that each file path exists on disk.

    Args:
        csv_file (str): Path to the combined CSV file.
        check_file_existence (bool): If True, check for file existence on disk.
    """
    # Load CSV into DataFrame
    df = pd.read_csv(csv_file)
    
    # Check for null values in "image_path"
    null_count = df['image_path'].isnull().sum()
    
    # Check for empty strings (after stripping whitespace)
    empty_count = (df['image_path'].astype(str).str.strip() == '').sum()
    
    if null_count > 0 or empty_count > 0:
        print(f"Found {null_count} null and {empty_count} empty file paths in the CSV.")
        missing_df = df[df['image_path'].isnull() | (df['image_path'].astype(str).str.strip() == '')]
        print("Rows with missing file paths:")
        print(missing_df)
    else:
        print("All file paths in the CSV are non-null and non-empty.")
    
    if check_file_existence:
        missing_files = []
        for idx, path in enumerate(df['image_path']):
            # Make sure we are working with a string.
            path = str(path).strip()
            if not os.path.exists(path):
                missing_files.append((idx, path))
        
        if missing_files:
            print("\nThe following file paths do not exist on disk:")
            for idx, path in missing_files:
                print(f"Row {idx}: {path}")
        else:
            print("\nVerified: All file paths exist on disk.")

if __name__ == "__main__":
    # Path to the combined CSV file created by the earlier script.
    csv_file = "E:\\Sentinelv3\\Combined_Forest_FilePaths.csv"
    test_file_paths(csv_file, check_file_existence=True)