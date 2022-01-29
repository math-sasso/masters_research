from pathlib import Path

from easy_sdm.data.data_loader import ShapefileLoader
from easy_sdm.featuarizer.dataset_builder import OccurrancesDatasetBuilder
from easy_sdm.utils.utils import PathUtils


def test_extract_occurances(mock_species_shapefile_path: Path):
    processed_rasters_dirpath = PathUtils.dir_path(Path.cwd() / "data/processed_rasters/standarized_rasters")
    species_shapefile_path = PathUtils.file_path(mock_species_shapefile_path)
    raster_paths_list = PathUtils.get_rasters_filepaths_in_dir(
        processed_rasters_dirpath
    )
    occ_dst_builder = OccurrancesDatasetBuilder(raster_paths_list)
    df = occ_dst_builder.build(
        ShapefileLoader(species_shapefile_path).load_dataset()
    )

    assert(df.empty is False)
    assert(df.index.names == ['lat', 'lon'])