from pathlib import Path

import numpy as np
import rasterio
from easy_sdm.data import RasterInfoExtractor, RasterLoader
from easy_sdm.data_colector import burn_shapefile_in_raster, standarize_rasters

def test_standarize_rasters(tmp_path, raw_rasters_dirpath):

    standarize_rasters(raw_rasters_dirpath, tmp_path)
    raster1 = tmp_path / "bio1_annual_mean_temperature.tif"
    raster2 = tmp_path / "envir1_annual_PET.tif"
    assert Path(raster1).is_file()
    assert Path(raster2).is_file()
    assert isinstance(RasterLoader(raster1).load_dataset(), rasterio.io.DatasetReader)


def test_burn_shapefile_in_raster(
    tmp_path, mock_map_shapefile_path, mock_processed_raster_path
):
    output_path = tmp_path / "brazilian_mask.tif"
    burn_shapefile_in_raster(
        reference_raster_path=mock_processed_raster_path,
        shapefile_path=mock_map_shapefile_path,
        output_path=output_path,
    )

    raster_mask = RasterLoader(output_path).load_dataset()
    raster_mask_array = RasterInfoExtractor(raster_mask).get_array()
    array_01 = np.array([-9999.0, 0])
    uniques = np.unique(raster_mask_array)
    assert np.array_equal(uniques, array_01)