from pathlib import Path

import rasterio
from easy_sdm.configs import configs
from easy_sdm.enums import RasterSource
from easy_sdm.utils import (PathUtils, RasterLoader, ShapefileLoader,
                            TemporaryDirectory)
from rasterio.plot import show

from .processing.raster_cliper import RasterCliper
from .processing.raster_data_standarizer import RasterDataStandarizer
from .processing.raster_shapefile_burner import RasterShapefileBurner


class RasterProcessingJob:
    """[A class that centralizes all RasterStandarization applications and take control over it]"""

    def __init__(self, processed_rasters_dir: Path, raw_rasters_dir: Path) -> None:
        self.configs = configs
        self.processed_rasters_dir = processed_rasters_dir
        self.environment_variables_rasters = (
            processed_rasters_dir / "environment_variables_rasters"
        )
        self.raw_rasters_dir = raw_rasters_dir
        self.__build_empty_folders()

    def __build_empty_folders(self):
        folder_list = [e.name for e in RasterSource]
        for folder in folder_list:
            PathUtils.create_folder(self.environment_variables_rasters / folder)

    def build_mask(self, shapefile_path: Path):
        reference_raster_path = (
            self.environment_variables_rasters
            / RasterSource.Bioclim.name
            / "bio1_annual_mean_temperature.tif"
        )
        output_path = self.processed_rasters_dir / "region_mask.tif"
        reference_raster_path = PathUtils.file_path(reference_raster_path)
        raster = RasterLoader(reference_raster_path).load_dataset()
        gdf = ShapefileLoader(shapefile_path).load_dataset()
        raster_bunner = RasterShapefileBurner(reference_raster=raster)
        raster_bunner.burn_and_save(shapfile=gdf, output_path=output_path)

    def __standarize_soilgrids(self, input_path: Path, output_path: Path):
        PathUtils.create_folder(output_path.parents[0])
        RasterDataStandarizer().standarize(
            raster=rasterio.open(input_path), output_path=output_path
        )

    def __standarize_bioclim_envirem(self, input_path: Path, output_path: Path):
        tempdir = TemporaryDirectory()
        raster_cliped_path = Path(tempdir.name) / "raster_cliped.tif"
        RasterCliper().clip_and_save(
            source_raster=rasterio.open(input_path), output_path=raster_cliped_path
        )
        PathUtils.create_folder(output_path.parents[0])
        RasterDataStandarizer().standarize(
            raster=rasterio.open(raster_cliped_path), output_path=output_path
        )

    def process_rasters_from_all_sources(self):
        self.process_rasters_from_source(RasterSource.Bioclim)
        self.process_rasters_from_source(RasterSource.Envirem)
        self.process_rasters_from_source(RasterSource.Soilgrids)

    def process_rasters_from_source(
        self, raster_source: RasterSource,
    ):
        source_dirpath = self.raw_rasters_dir / raster_source.name
        destination_dirpath = self.environment_variables_rasters / raster_source.name

        for filepath, filename in zip(
            PathUtils.get_rasters_filepaths_in_dir(source_dirpath),
            PathUtils.get_rasters_filenames_in_dir(source_dirpath),
        ):
            output_path = destination_dirpath / filename
            if not Path(output_path).is_file():
                if raster_source in [RasterSource.Envirem, RasterSource.Bioclim]:
                    self.__standarize_bioclim_envirem(
                        input_path=filepath, output_path=output_path,
                    )
                elif raster_source == RasterSource.Soilgrids:
                    self.__standarize_soilgrids(
                        input_path=filepath, output_path=output_path,
                    )
            else:
                print(f"Raster {filename} already exists")
