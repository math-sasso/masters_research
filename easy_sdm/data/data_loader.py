from pathlib import Path
from numpy import array

from numpy.lib.shape_base import take_along_axis
import rasterio
import geopandas as gpd
import fiona
from configs import configs

class RasterLoader:
    def __init__(self):
        self.configs = configs

    def read(self, raster_path: Path):
        raster = rasterio.open(raster_path)
        #raster = self.__read_and_check(raster)
        return raster


class ShapefileLoader:
    def __init__(self):
        ...

    def read_geodataframe(self, shapefile_path: Path):
        shp = gpd.read_file(shapefile_path)
        return shp

    def read_shapes(self, shapefile_path: Path):
        with fiona.open("tests/data/box.shp", "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        return shapes
