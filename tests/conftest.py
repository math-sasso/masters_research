from pathlib import Path

import pytest
from easy_sdm.utils import PathUtils

############################
#           Paths          #
############################

@pytest.fixture
def raw_rasters_dirpath():
    raw_rasters_dirpath = PathUtils.dir_path(Path.cwd() / "data/raw/rasters")
    return raw_rasters_dirpath


@pytest.fixture
def processed_rasters_dirpath():
    processed_rasters_dirpath = PathUtils.dir_path(
        Path.cwd() / "data/processed_rasters/standarized_rasters"
    )
    return processed_rasters_dirpath


@pytest.fixture
def mock_species_shapefile_path():
    path = Path.cwd() / "data/species_data/occurances/Cajanus_cajan/Cajanus_cajan.shp"
    return path


@pytest.fixture
def mock_map_shapefile_path():
    path = Path.cwd() / "data/raw/shapefiles_brasil/level_0/BRA_adm0.shp"
    return path

@pytest.fixture
def mock_processed_raster_path():
    path = (
        Path.cwd()
        / "data/processed_rasters/standarized_rasters/bio1_annual_mean_temperature.tif"
    )
    return path


@pytest.fixture
def mock_raw_raster_path():
    path = (
        Path.cwd() / "data/raw/rasters/Bioclim_Rasters/bio1_annual_mean_temperature.tif"
    )
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


@pytest.fixture
def mock_species_shapefile_dataloader(mock_species_shapefile_path):
    from easy_sdm.data.data_loader import ShapefileLoader

    class MockSpeciesShapefileLoader:
        def setup(self):
            pass

        def load_dataset(self):

            dataloader = ShapefileLoader(mock_species_shapefile_path)
            shp = dataloader.load_dataset()
            return shp

    return MockSpeciesShapefileLoader()


@pytest.fixture
def mock_map_shapefile_dataloader(mock_map_shapefile_path):
    from easy_sdm.data.data_loader import ShapefileLoader

    class MockMapShapefileLoader:
        def setup(self):
            pass

        def load_dataset(self):

            dataloader = ShapefileLoader(mock_map_shapefile_path)
            shp = dataloader.load_dataset()
            return shp

    return MockMapShapefileLoader()


@pytest.fixture
def mock_processed_raster_dataloader(mock_processed_raster_path):
    from easy_sdm.data.data_loader import RasterLoader

    class MockProcessedRasterLoader:
        def setup(self):
            pass

        def load_dataset(self):
            dataloader = RasterLoader(mock_processed_raster_path)
            shp = dataloader.load_dataset()
            return shp

    return MockProcessedRasterLoader()


@pytest.fixture
def mock_raw_raster_dataloader(mock_raw_raster_path):
    from easy_sdm.data.data_loader import RasterLoader

    class MockRawRasterLoader:
        def setup(self):
            pass

        def load_dataset(self):
            dataloader = RasterLoader(mock_raw_raster_path)
            shp = dataloader.load_dataset()
            return shp

    return MockRawRasterLoader()