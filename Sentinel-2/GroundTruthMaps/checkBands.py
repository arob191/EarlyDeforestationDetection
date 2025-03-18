import rasterio

image_path = 'E:\Sentinelv3\Fazenda Forest\Fazenda_Manna_2015_2016\Fazenda_Manna_2015_2016_Tile_001.tif'  # Directory containing your CSV files


with rasterio.open(image_path) as src:
    print(f"Number of bands: {src.count}")
    print(f"Band descriptions: {src.descriptions}")
    print(f"Indexes: {src.indexes}")
