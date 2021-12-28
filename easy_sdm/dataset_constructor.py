from pathlib import Path

from easy_sdm.data import ShapefileLoader
from easy_sdm.featuarizer import OccurrancesDatasetBuilder
from easy_sdm.utils import PathUtils


def extract_occurances(species_shapefile_path: Path):
    processed_rasters_dirpath = PathUtils.dir_path(
        Path.cwd() / "data/processed_rasters/standarized_rasters"
    )
    species_shapefile_path = PathUtils.file_path(species_shapefile_path)
    raster_paths_list = PathUtils.get_rasters_filepaths_in_dir(
        processed_rasters_dirpath
    )
    occ_dst_builder = OccurrancesDatasetBuilder(raster_paths_list)
    df = occ_dst_builder.build(ShapefileLoader(species_shapefile_path).load_dataset())
    import pdb

    pdb.set_trace()
    return df


extract_occurances(
    Path.cwd() / "data/species_data/occurances/Cajanus_cajan/Cajanus_cajan.shp"
)
