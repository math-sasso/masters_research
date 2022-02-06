from pathlib import Path
from turtle import width
import rasterio
import geopandas as gpd

from easy_sdm.data import (
    RasterCliper,
    RasterLoader,
    RasterShapefileBurner,
    RasterStandarizer,
    ShapefileLoader,
    ShapefileRegion,
    Species,
    SpeciesGDFBuilder,
    SoilgridsDownloader,
)
from easy_sdm.utils import PathUtils

shp_region = ShapefileRegion(
    Path.cwd() / "data/raw/shapefiles_brasil/level_0/BRA_adm0.shp"
)


def download_soigrids_all_rasters(coverage_filter: str):
    variables = [
        "wrb",
        "cec",
        "clay",
        "phh2o",
        "silt",
        "ocs",
        "bdod",
        "cfvo",
        "nitrogen",
        "sand",
        "soc",
        "ocd",
    ]

    for variable in variables:
        download_soilgrods_one_raster(variable, coverage_filter)


def download_soilgrods_one_raster(variable: str, coverage_filter: str):
    reference_raster_path = (
        Path.cwd()
        / "data"
        / "processed_rasters"
        / "standarized_rasters"
        / "Bioclim_Rasters"
        / "bio1_annual_mean_temperature.tif"
    )
    root_dir = Path.cwd() / "data" / "raw" / "rasters" / "Soilgrids_Rasters"
    soilgrids_downloader = SoilgridsDownloader(
        reference_raster_path=reference_raster_path, root_dir=root_dir
    )
    soilgrids_downloader.set_soilgrids_requester(variable=variable)
    coverages = soilgrids_downloader.get_coverage_list()
    for cov in coverages:
        if coverage_filter in cov:
            soilgrids_downloader.download(coverage_type=cov)
    # filtered_coverages = [cov for cov in coverages if coverage_filter in cov]


def standarize_rasters(source_dirpath: str, destination_dirpath: str, raster_type: str):

    raster_standarizer = RasterStandarizer()
    for filepath, filename in zip(
        PathUtils.get_rasters_filepaths_in_dir(source_dirpath),
        PathUtils.get_rasters_filenames_in_dir(source_dirpath),
    ):
        try:
            if raster_type in ["bioclim", "elevation", "envirem"]:
                raster_standarizer.standarize_bioclim_envirem(
                    input_path=filepath, output_path=destination_dirpath / filename,
                )
            elif raster_type == "soilgrids":
                raster_standarizer.standarize_soilgrids(
                    input_path=filepath, output_path=destination_dirpath / filename,
                )

        except FileExistsError as exc:
            print(str(exc))
            print(f"filepath: {filepath}")


# standarize_rasters(Path.cwd()/ "data/raw/rasters/Bioclim_Rasters",Path.cwd()/ "data/processed_rasters/standarized_rasters/Bioclim_Rasters","bioclim")


def burn_shapefile_in_raster(
    reference_raster_path: Path, shapefile_path: Path, output_path: Path
):
    reference_raster_path = PathUtils.file_path(reference_raster_path)
    raster = RasterLoader(reference_raster_path).load_dataset()
    gdf = ShapefileLoader(shapefile_path).load_dataset()
    raster_bunner = RasterShapefileBurner(raster)
    raster_bunner.burn_and_save(shapfile=gdf, output_path=output_path)


# burn_shapefile_in_raster(reference_raster_path=Path.cwd() / "data/processed_rasters/standarized_rasters/Bioclim_Rasters/bio1_annual_mean_temperature.tif",
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
