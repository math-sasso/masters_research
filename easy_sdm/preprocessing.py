from pathlib import Path

import geopandas as gpd

from data.data_loader import RasterLoader,ShapefileLoader
from data.raster_processors import RasterCliper, RasterShapefileBurner
from utils.utils import PathUtils

raster_loader = RasterLoader()
shapefile_loader = ShapefileLoader()
raster_cliper = RasterCliper()


def standarize_rasters(source_dirpath: str, destination_dirpath: str):
    # source_dirpath: Path.cwd() / 'data/raw'
    # destination_dirpath: Path.cwd() / 'data/standarized_rasters'
    for filepath, filename, varname in PathUtils.get_rasters_in_folder(source_dirpath):
        try:
            raster = raster_loader.read(filepath)
            raster_cliper.clip_and_save(
                source_raster=raster,
                output_path=PathUtils.file_path_existis(destination_dirpath / filename),
            )
        except FileExistsError as exc:
            print(str(exc))
            print(f"filepath: {filepath}")


# standarize_rasters(Path.cwd()/ "data/raw",Path.cwd()/ "data/standarized_rasters")


def burn_shapefile_in_raster(reference_raster_path: str, shapefile_path:str, output_path:str):
    reference_raster_path = PathUtils.file_path(reference_raster_path)
    raster = raster_loader.read(reference_raster_path)
    gdf = shapefile_loader.read_geodataframe(shapefile_path)
    raster_bunner = RasterShapefileBurner(raster)
    raster_bunner.burn_and_save(shapfile=gdf,output_path=output_path)
    



burn_shapefile_in_raster(reference_raster_path=Path.cwd() / "data/processed_rasters/standarized_rasters/bio1_annual_mean_temperature.tif", 
                        shapefile_path=Path.cwd() / "data/raw/shapefiles_brasil/level_0/BRA_adm0.shp",
                        output_path=Path.cwd() / "data/processed_rasters/others/brazilian_mask.tif")