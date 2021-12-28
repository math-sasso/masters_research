from pathlib import Path

import numpy as np
import rasterio
from easy_sdm.data import RasterInfoExtractor, RasterLoader, RasterCliper
from easy_sdm.utils import PathUtils

def test_clip_raster(tmp_path, raw_rasters_dirpath):

    filepath = PathUtils.get_rasters_filepaths_in_dir(raw_rasters_dirpath)[0]
    filename = PathUtils.get_rasters_filenames_in_dir(raw_rasters_dirpath)[0]
    output_path = tmp_path / filename
    raster = RasterLoader(filepath).load_dataset()
    RasterCliper().clip_and_save(
        source_raster=raster,
        output_path=PathUtils.file_path_existis(output_path),
    )
    assert Path(output_path).is_file()
    assert isinstance(RasterLoader(output_path).load_dataset(), rasterio.io.DatasetReader)

