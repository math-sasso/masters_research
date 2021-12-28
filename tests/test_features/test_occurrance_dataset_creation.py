from pathlib import Path

from easy_sdm.data.data_loader import ShapefileLoader
from easy_sdm.featuarizer.build_features import OccurrancesDatasetBuilder
from easy_sdm.utils.utils import PathUtils


def extract_occurances(species_shapefile_path: Path):
    processed_rasters_dirpath = PathUtils.dir_path(Path.cwd() / "data/processed_rasters/standarized_rasters")
    species_shapefile_path = PathUtils.file_path(species_shapefile_path)
    raster_paths_list = PathUtils.get_rasters_filepaths_in_dir(
        processed_rasters_dirpath
    )
    occ_dst_builder = OccurrancesDatasetBuilder(raster_paths_list)
    df = occ_dst_builder.build(
        ShapefileLoader().read_geodataframe(species_shapefile_path)
    )
    assert(df.shape[0]>0)
    assert(df.index.names == ['lat', 'lon'])