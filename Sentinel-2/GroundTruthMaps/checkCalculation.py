import matplotlib.pyplot as plt
import rasterio

# Load NDVI difference and ternary mask
with rasterio.open("E:\\Sentinelv3\\NDVI_Outputs\\Iracema\\1_NDVI_Diff_2021_2022_to_2023_2024.tif") as ndvi_diff_src:
    ndvi_diff = ndvi_diff_src.read(1)

with rasterio.open("E:\\Sentinelv3\\NDVI_Outputs\\Iracema\\1_Ternary_Mask_2021_2022_to_2023_2024.tif") as ternary_mask_src:
    ternary_mask = ternary_mask_src.read(1)

# Plot NDVI difference and ternary mask
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('NDVI Difference')
plt.imshow(ndvi_diff, cmap='RdYlGn')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title('Ternary Mask')
plt.imshow(ternary_mask, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()

plt.show()