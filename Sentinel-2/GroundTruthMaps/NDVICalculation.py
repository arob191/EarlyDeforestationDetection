import os
import rasterio
import pandas as pd
import numpy as np
import csv
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

def calculate_ndvi_from_multiband(image_path, output_path):
    with rasterio.open(image_path) as src:
        red = src.read(1).astype('float32')  # B4 (Red)
        nir = src.read(4).astype('float32')  # B8 (NIR)

        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nir - red) / (nir + red)
        ndvi = np.where((nir + red) == 0, 0, ndvi)

        meta = src.meta
        meta.update(dtype='float32', count=1)
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(ndvi, 1)

    return ndvi

def process_tile(row, output_dir, time_period):
    tile_id = row['tile_id']
    image_path = row['image_path']
    output_path = os.path.join(output_dir, f"{tile_id}_NDVI_{time_period}.tif")

    if os.path.exists(image_path):
        return tile_id, calculate_ndvi_from_multiband(image_path, output_path)
    else:
        print(f"File not found: {image_path}")
        return tile_id, None

def process_tiles(csv_file, output_dir, time_period):
    df = pd.read_csv(csv_file)
    ndvi_results = {}

    with Pool() as pool:
        results = pool.starmap(
            process_tile, [(row, output_dir, time_period) for _, row in df.iterrows()]
        )
        for tile_id, ndvi in results:
            if ndvi is not None:
                ndvi_results[tile_id] = ndvi

    print(f"NDVI calculation complete for {time_period}.")
    return ndvi_results

def calculate_ndvi_difference_and_deforestation(ndvi_dict1, ndvi_dict2, output_dir, time_period1, time_period2, ndvi_threshold=-0.2):
    deforestation_data = []

    for tile_id in ndvi_dict1:
        if tile_id in ndvi_dict2:
            ndvi_diff = ndvi_dict2[tile_id] - ndvi_dict1[tile_id]
            output_file = os.path.join(output_dir, f"{tile_id}_NDVI_Diff_{time_period1}_to_{time_period2}.tif")

            with rasterio.open(output_file, "w", driver='GTiff', height=ndvi_diff.shape[0], width=ndvi_diff.shape[1], count=1, dtype='float32') as dst:
                dst.write(ndvi_diff, 1)

            total_pixels = ndvi_diff.size
            deforested_pixels = np.sum(ndvi_diff < ndvi_threshold)
            deforestation_percentage = (deforested_pixels / total_pixels) * 100

            deforestation_data.append({
                "tile_id": tile_id,
                "time_period": f"{time_period1}_to_{time_period2}",
                "deforestation_percentage": deforestation_percentage,
                "ndvi_diff_path": output_file
            })

            # Generate a heatmap for this tile
            plt.figure(figsize=(10, 6))
            sns.heatmap(ndvi_diff, cmap='coolwarm', center=0)
            plt.title(f"NDVI Difference Heatmap: {tile_id} ({time_period1} to {time_period2})")
            plt.savefig(os.path.join(output_dir, f"{tile_id}_NDVI_Heatmap_{time_period1}_to_{time_period2}.png"))
            plt.close()

    print(f"NDVI difference calculated and deforestation data collected for {time_period1} to {time_period2}.")
    return deforestation_data

def save_deforestation_csv(output_dir, deforestation_data):
    csv_file = os.path.join(output_dir, "deforestation_data.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["tile_id", "time_period", "deforestation_percentage", "ndvi_diff_path"])
        for entry in deforestation_data:
            writer.writerow([entry["tile_id"], entry["time_period"], entry["deforestation_percentage"], entry["ndvi_diff_path"]])
    print(f"Deforestation data CSV saved to: {csv_file}")

def validate_with_ground_truth(deforestation_data, ground_truth_path):
    print("Validation placeholder: Integrate ground truth comparison here.")
    # Placeholder for incorporating validation logic

if __name__ == '__main__':
    base_csv_dir = 'E:\\Sentinelv3\\Fazenda Forest'
    output_directory = 'E:\\Sentinelv3\\Fazenda Forest\\NDVI_Outputs'
    forest = "Fazenda"
    time_periods = ["2015_2016", "2017_2018", "2019_2020", "2021_2022", "2023_2024"]

    ndvi_data = {}
    for period in time_periods:
        csv_path = os.path.join(base_csv_dir, f"{forest}_{period}.csv")
        ndvi_data[period] = process_tiles(csv_path, output_directory, period)

    all_deforestation_data = []
    all_deforestation_data.extend(calculate_ndvi_difference_and_deforestation(
        ndvi_data["2015_2016"], ndvi_data["2017_2018"], output_directory, "2015_2016", "2017_2018"))
    all_deforestation_data.extend(calculate_ndvi_difference_and_deforestation(
        ndvi_data["2017_2018"], ndvi_data["2019_2020"], output_directory, "2017_2018", "2019_2020"))
    all_deforestation_data.extend(calculate_ndvi_difference_and_deforestation(
        ndvi_data["2019_2020"], ndvi_data["2021_2022"], output_directory, "2019_2020", "2021_2022"))
    all_deforestation_data.extend(calculate_ndvi_difference_and_deforestation(
        ndvi_data["2021_2022"], ndvi_data["2023_2024"], output_directory, "2021_2022", "2023_2024"))

    save_deforestation_csv(output_directory, all_deforestation_data)
    # validate_with_ground_truth(all_deforestation_data, "path_to_ground_truth.csv")
