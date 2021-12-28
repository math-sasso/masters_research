from pathlib import Path

import geopandas as gpd
import rasterio
from easy_sdm.configs import configs


class RasterLoader:
    def __init__(self, raster_path: Path):
        self.configs = configs
        self.raster_path = raster_path

    def load_dataset(self):
        raster = rasterio.open(self.raster_path)
        # raster = self.__read_and_check(raster)
        return raster


class ShapefileLoader:
    def __init__(self, shapefile_path: Path):
        self.shapefile_path = shapefile_path

    def load_dataset(self):
        shp = gpd.read_file(self.shapefile_path)
        return shp
