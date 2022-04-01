from pathlib import Path

import pandas as pd
import pytest
from easy_sdm.utils import PathUtils

############################
#           Paths          #
############################

# @pytest.fixture
# def raw_rasters_dirpath():
#     raw_rasters_dirpath = PathUtils.dir_path(Path.cwd() / "data/download/raw_rasters")
#     return raw_rasters_dirpath


# @pytest.fixture
# def processed_rasters_dirpath():
#     processed_rasters_dirpath = PathUtils.dir_path(
#         Path.cwd() / data/raster_processing/environment_variables_rasters
#     )
#     return processed_rasters_dirpath


@pytest.fixture
def mock_species_shapefile_path():
    path = Path.cwd() / "extras/mock_species"
    return path


@pytest.fixture
def mock_map_shapefile_path():
    path = Path.cwd() / "extras/mock_region_region_shapefile"
    return path


@pytest.fixture
def mock_processed_raster_path():
    path = Path.cwd() / "extras/mock_processed_raster.tif"
    return path


@pytest.fixture
def mock_raw_raster_path():
    path = Path.cwd() / "extras/mock_raw_raster.tif"
    return path


@pytest.fixture
def mock_mask_raster_path():
    path = Path.cwd() / "extras/mock_region_mask.tif"
    return path


############################
#           Lists          #
############################


@pytest.fixture
def processed_raster_paths_list(processed_rasters_dirpath):
    raster_paths_list = PathUtils.get_rasters_filepaths_in_dir(
        processed_rasters_dirpath
    )
    return raster_paths_list


@pytest.fixture
def raw_raster_paths_list(raw_rasters_dirpath):
    raster_paths_list = PathUtils.get_rasters_filepaths_in_dir(raw_rasters_dirpath)
    return raster_paths_list


############################
#           Special          #
############################

# O certo seria deixar isso aqui e criar um MockRasterStatisticsCalculator
# @pytest.fixture
# def df_stats(tmp_path, mock_mask_raster_path,processed_raster_paths_list):

#     from easy_sdm.dataset_creation import RasterStatisticsCalculator

#     output_path = tmp_path / "rasters_statistics.csv"
#     RasterStatisticsCalculator(
#         raster_path_list=processed_raster_paths_list, mask_raster_path=mock_mask_raster_path
#     ).build_table(output_path)

#     df_stats = pd.read_csv(output_path)
#     df_stats.to_csv('extras/rasters_statistics.csv',index=False)
#     return df_stats
