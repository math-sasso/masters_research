from easy_sdm.utils import PathUtils, RasterLoader
from easy_sdm.raster_processing import RasterCliper
import rasterio
from pathlib import Path


def test_clip_raster(tmp_path, mock_raw_raster_path):

    output_path = tmp_path / "clipped_raster.tif"
    raster = RasterLoader(mock_raw_raster_path).load_dataset()
    RasterCliper().clip_and_save(
        source_raster=raster,
        output_path=PathUtils.file_path_existis(output_path),
    )
    loaded_raster = RasterLoader(output_path).load_dataset()
    assert Path(output_path).is_file()
    assert isinstance(loaded_raster, rasterio.io.DatasetReader)
    assert loaded_raster.shape[0] > loaded_raster.shape[1]
