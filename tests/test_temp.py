from distutils.command.config import config
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from easy_sdm.data import RasterStandarizer, ShapefileLoader
from easy_sdm.data_colector import burn_shapefile_in_raster, standarize_rasters
from easy_sdm.featuarizer import (
    OccurrancesDatasetBuilder,
    SDMDatasetCreator,
    RasterStatisticsCalculator,
)
from easy_sdm.utils import PathUtils
from easy_sdm.configs import configs


def test_statistics_table_generatio(tmp_path,mock_master_raster_path):
    processed_rasters_dirpath = PathUtils.dir_path(
        Path.cwd() / "data/processed_rasters/standarized_rasters"
    )
    raster_path_list = PathUtils.get_rasters_filepaths_in_dir(processed_rasters_dirpath)

    output_path = tmp_path / 'rasters_statistics.csv'
    RasterStatisticsCalculator(
        raster_path_list=raster_path_list, mask_raster_path=mock_master_raster_path
    ).build_table(output_path)

    df_stats = pd.read_csv(output_path)
    import pdb;pdb.set_trace()


    # def test_sdm_dataset_creator(mock_species_shapefile_path):
    #     processed_rasters_dirpath = PathUtils.dir_path(Path.cwd() / "data/processed_rasters/standarized_rasters")
    #     species_shapefile_path = PathUtils.file_path(mock_species_shapefile_path)
    #     SDMDatasetCreator(raster_path_list=, statistics_dataset=)

    #     class SDMDatasetCreator:
    #     """[Create a dataset with species and pseudo spescies for SDM Machine Learning]
    #     """

    #     def __init__(
    #         self, raster_path_list: List[Path], statistics_dataset: pd.DataFrame
    #     ) -> None:
    #         # ps_generator: BasePseudoSpeciesGenerator
    #         # self.ps_generator = ps_generator
    #         self.statistics_dataset = statistics_dataset
    #         self.occ_dataset_builder = OccurrancesDatasetBuilder(raster_path_list)

    # raster_paths_list = PathUtils.get_rasters_filepaths_in_dir(
    #     processed_rasters_dirpath
    # )
    # occ_dst_builder = OccurrancesDatasetBuilder(raster_paths_list)
    # df = occ_dst_builder.build(
    #     ShapefileLoader(species_shapefile_path).load_dataset()
    # )

    # df.to_csv(Path.cwd() / 'extras' / "mock_occurrences.csv" ,index=False)

    SDMDatasetCreator()
