import pandas as pd

# Load the CSV file into a data frame
df = pd.read_csv('E:\\Fazenda.csv')

# Base path and file name pattern
base_path = 'E:\\Sentinelv3\\Fazenda Forest\\Fazenda_Manna_2019_2020\\Fazenda_Manna_2019_2020_Tile_'
file_extension = '.tif'

# Create a list to store the dynamic values
dynamic_values = [f"{base_path}{i+1:03d}{file_extension}" for i in range(100)]

# Ensure the data frame has at least 100 rows
if len(df) < 100:
    additional_rows = pd.DataFrame({df.columns[0]: [None] * (100 - len(df))})
    df = pd.concat([df, additional_rows], ignore_index=True)

# Update the first column of the specified number of rows with dynamic values
df.iloc[:100, 1] = dynamic_values

for i in range(1, 101):
    df.iloc[i - 1, 0] = i

# Print the first few rows to verify the update
print(df.head(105))

# Save the updated data frame back to a new CSV file
output_path = 'C:\\Users\\Austin\\OneDrive\\Documents\\Personal Projects\\GitHub\\EarlyDeforestationDetection\\Fazenda_2019_2020.csv'
df.to_csv(output_path, index=False)

print(f"Updated CSV file saved to: {output_path}")

