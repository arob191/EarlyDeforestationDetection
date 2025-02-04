import rasterio
from rasterio.enums import Resampling
import numpy as np

# Function to normalize and cast data to uint8
def normalize_to_uint8(data):
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    normalized_data = (data - min_val) / (max_val - min_val) * 255
    return normalized_data.astype(np.uint8)

# Open the GeoTIFF file
with rasterio.open('E:\Amazon\AmazonForest_images_2015-01 (1).tif') as src:
    # Read the RGB bands (assuming bands 4, 3, 2)
    rgb_data = src.read([4, 3, 2], out_shape=(3, int(src.height), int(src.width)), resampling=Resampling.bilinear)
    
    # Normalize and cast the data to uint8
    rgb_data_uint8 = normalize_to_uint8(rgb_data)

# Save the RGB bands as a PNG file
with rasterio.open('E:\\Amazon\\AmazonForest_images_2015-01.png', 'w', driver='PNG',
                   height=rgb_data_uint8.shape[1], width=rgb_data_uint8.shape[2],
                   count=3, dtype=rgb_data_uint8.dtype) as dst:
    dst.write(rgb_data_uint8)

print('PNG export complete')