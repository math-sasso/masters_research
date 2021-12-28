from pathlib import Path

import geopandas as gpd

from easy_sdm.data import (
    RasterCliper,
    RasterLoader,
    RasterShapefileBurner,
    ShapefileLoader,
    ShapefileRegion,
    Species,
    SpeciesGDFBuilder,
)
from easy_sdm.utils import PathUtils

shp_region = ShapefileRegion(
    Path.cwd() / "data/raw/shapefiles_brasil/level_0/BRA_adm0.shp"
)


def standarize_rasters(source_dirpath: str, destination_dirpath: str):

    for filepath, filename in zip(
        PathUtils.get_rasters_filepaths_in_dir(source_dirpath),
        PathUtils.get_rasters_filenames_in_dir(source_dirpath),
    ):
        try:
            raster = RasterLoader(filepath).load_dataset()
            RasterCliper().clip_and_save(
                source_raster=raster,
                output_path=PathUtils.file_path_existis(destination_dirpath / filename),
            )
        except FileExistsError as exc:
            print(str(exc))
            print(f"filepath: {filepath}")


# standarize_rasters(Path.cwd()/ "data/raw",Path.cwd()/ "data/standarized_rasters")


def burn_shapefile_in_raster(
    reference_raster_path: Path, shapefile_path: Path, output_path: Path
):
    reference_raster_path = PathUtils.file_path(reference_raster_path)
    raster = RasterLoader(reference_raster_path).load_dataset()
    gdf = ShapefileLoader(shapefile_path).load_dataset()
    raster_bunner = RasterShapefileBurner(raster)
    raster_bunner.burn_and_save(shapfile=gdf, output_path=output_path)


# burn_shapefile_in_raster(reference_raster_path=Path.cwd() / "data/processed_rasters/standarized_rasters/bio1_annual_mean_temperature.tif",
#                         shapefile_path=Path.cwd() / "data/raw/shapefiles_brasil/level_0/BRA_adm0.shp",
#                         output_path=Path.cwd() / "data/processed_rasters/others/brazilian_mask.tif")


def collect_species_data(
    species_id: int, species_name: str, shp_region: ShapefileRegion, output_dir: Path
):
    species_gdf_builder = SpeciesGDFBuilder(
        Species(taxon_key=species_id, name=species_name), shp_region
    )
    species_name = species_name.replace(" ", "_")
    path = output_dir / species_name / f"{species_name}.shp"
    species_gdf_builder.save_species_gdf(path)
