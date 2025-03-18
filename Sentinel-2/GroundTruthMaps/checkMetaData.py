import rasterio


with rasterio.open("E:/Sentinelv3/Fazenda Forest/NDVI_Outputs/1_NDVI_2019_2020.tif") as src:
    print(f"CRS: {src.crs}")
    print(f"Transform: {src.transform}")

with rasterio.open('E:\\Sentinelv3\\Fazenda Forest\\Fazenda_Manna_2015_2016\\Fazenda_Manna_2015_2016_Tile_001.tif') as src:
    print(f"CRS: {src.crs}")
    print(f"Transform: {src.transform}")