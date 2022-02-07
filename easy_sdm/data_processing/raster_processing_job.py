from pathlib import Path

import rasterio
from easy_sdm.configs import configs
from utils import PathUtils

from .environment_data.raster_shapefile_burner import RasterShapefileBurner
from .environment_data.raster_data_standarizer import RasterDataStandarizer
from .environment_data.raster_cliper import RasterCliper

from easy_sdm.utils import RasterLoader, ShapefileLoader, TemporaryDirectory


class RasterProcessingJob:
    """[A class that centralizes all RasterStandarization applications and take control over it]"""

    def __init__(self) -> None:
        self.configs = configs

    def build_mask(
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

    def standarize_soilgrids(self, input_path: Path, output_path: Path):
        PathUtils.create_folder(output_path.parents[0])
        RasterDataStandarizer().standarize(
            raster=rasterio.open(input_path), output_path=output_path
        )

    def standarize_bioclim_envirem(self, input_path: Path, output_path: Path):
        tempdir = TemporaryDirectory()
        raster_cliped_path = Path(tempdir.name) / "raster_cliped.tif"
        RasterCliper().clip_and_save(
            source_raster=rasterio.open(input_path), output_path=raster_cliped_path
        )
        PathUtils.create_folder(output_path.parents[0])
        RasterDataStandarizer().standarize(
            raster=rasterio.open(raster_cliped_path), output_path=output_path
        )

    def standarize_rasters(
        source_dirpath: str, destination_dirpath: str, raster_type: str
    ):

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
