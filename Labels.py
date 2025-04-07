import pandas as pd
import os

# Base directory containing all the forest folders
base_dir = r'E:\Sentinelv3'

# Configuration for each forest: key is the forest folder name and value is the naming pattern.
forest_config = {
    'Rio Aruana Forest': '{forest_base}_{time_period}\\{forest_base}_{time_period}_Tile_',
    'Fazenda Forest': '{forest_base}_Manna_{time_period}\\{forest_base}_Manna_{time_period}_Tile_',
    'Para Forest': '{forest_base}_{time_period}\\{forest_base}_{time_period}_Tile_',
    'Braunlage Forest': '{forest_base}_{time_period}\\{forest_base}_{time_period}_Tile_',
    'Cariboo Forest': '{forest_base}_{time_period}\\{forest_base}_{time_period}_Tile_',
    'Fort McMurray Forest': '{forest_base}_{time_period}\\{forest_base}_{time_period}_Tile_',
    'Sam Houston Forest': '{forest_base}_{time_period}\\{forest_base}_{time_period}_Tile_',
    'Oblast Forest': '{forest_base}_{time_period}\\{forest_base}_{time_period}_Tile_',
    'Tonkino Forest': '{forest_base}_{time_period}\\{forest_base}_{time_period}_Tile_',
    'Iracema Forest': '{forest_base}_{time_period}\\{forest_base}_{time_period}_Tile_'
}

# Time periods for which file paths are needed
time_periods = ['2015_2016', '2017_2018', '2019_2020', '2021_2022', '2023_2024']

# Base file extension for the image files
file_extension = '.tif'

# List to collect rows of data
rows = []

# Loop through each forest and time period, and build the file paths.
for forest, naming_pattern in forest_config.items():
    # Clean up the forest base name by replacing spaces and removing the word 'Forest'
    forest_base = forest.replace(' ', '_').replace('_Forest', '')
    forest_path = os.path.join(base_dir, forest)
    
    for time_period in time_periods:
        # Construct the subfolder path using the naming pattern
        subfolder = naming_pattern.format(forest_base=forest_base, time_period=time_period)
        base_path = os.path.join(forest_path, subfolder)
        
        # For each tile (assumed to be 1 to 100), create the full image file path.
        for tile_id in range(1, 101):
            image_path = f"{base_path}{tile_id:03d}{file_extension}"
            
            # Append each row as a dictionary.
            rows.append({
                'forest': forest,
                'forest_base': forest_base,
                'time_period': time_period,
                'tile_id': tile_id,
                'image_path': image_path
            })

# Create a DataFrame from the collected rows.
combined_df = pd.DataFrame(rows)

# Define the output CSV file path.
output_csv = os.path.join(base_dir, "Combined_Forest_FilePaths.csv")
combined_df.to_csv(output_csv, index=False)
print(f"Combined CSV file saved to: {output_csv}")