import rasterio
import geopandas as gpd
import fiona


class RasterLoader:
    def __init__(self):
        pass

    def read(cls, raster_path: str):
        raster = rasterio.open(raster_path)
        return raster


class ShapefileLoader:
    def __init__(self):
        ...

    def read_geodataframe(self,shapefile_path: str):
        shp = gpd.read_file(shapefile_path)
        return shp

    def read_shapes(self,shapefile_path: str):
        with fiona.open("tests/data/box.shp", "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        return shapes
