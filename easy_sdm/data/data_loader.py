import rasterio
import geopandas as gpd

class RasterLoader():
    def __init__(self):
        pass
    def read(self,raster_path:str):
        raster=rasterio.open(raster_path)
        return raster


class ShapefileLoader():
    def __init__(self):
        ...
    def read(shapefile_path:str):
        shp = gpd.read_file(shapefile_path)
        return shp