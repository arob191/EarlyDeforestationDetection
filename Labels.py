import pandas as pd
import os

# Base directory containing the forest folders
base_dir = 'E:\\Sentinelv3'

# Configuration for each forest directory and their corresponding naming conventions
forest_config = {
    'Rio Aruana Forest': '{forest_base}_{time_period}\\{forest_base}_{time_period}_Tile_',
    'Fazenda Forest': '{forest_base}_Manna_{time_period}\\{forest_base}_Manna_{time_period}_Tile_',
    'Para Forest': '{forest_base}_{time_period}\\{forest_base}_{time_period}_Tile_',
    'Braunlage Forest': '{forest_base}_{time_period}\\{forest_base}_{time_period}_Tile_',
    'Cariboo Forest': '{forest_base}_{time_period}\\{forest_base}_{time_period}_Tile_',
    'Fort McMurray Forest': '{forest_base}_{time_period}\\{forest_base}_{time_period}_Tile_',
    'Sam Houston Forest': '{forest_base}_{time_period}\\{forest_base}_{time_period}_Tile_'
}

# Time periods for which CSV files are needed
time_periods = ['2015_2016', '2017_2018', '2019_2020', '2021_2022', '2023_2024']

# Base file extension for .tif files
file_extension = '.tif'

# Loop through each forest and time period
for forest, naming_pattern in forest_config.items():
    forest_base = forest.replace(' ', '_').replace('_Forest', '')  # Extract the cleaned base name of the forest
    forest_path = os.path.join(base_dir, forest)

    for time_period in time_periods:
        # Construct dynamic file paths using the naming pattern
        subfolder = naming_pattern.format(
            forest_base=forest_base, time_period=time_period
        )
        base_path = os.path.join(forest_path, subfolder)
        csv_output_path = os.path.join(forest_path, f"{forest_base}_{time_period}.csv")

        # Create a DataFrame with 100 rows
        tile_ids = list(range(1, 101))  # Tile IDs from 1 to 100
        file_paths = [f"{base_path}{i:03d}{file_extension}" for i in tile_ids]
        df = pd.DataFrame({'tile_id': tile_ids, 'image_path': file_paths})

        # Save the updated DataFrame as a CSV file for this forest and time period
        df.to_csv(csv_output_path, index=False)
        print(f"Updated CSV file saved to: {csv_output_path}")