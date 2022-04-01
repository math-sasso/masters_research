from pathlib import Path


from pathlib import Path

from easy_sdm.featuarizer import OccurrancesDatasetBuilder
from easy_sdm.utils import ShapefileLoader


def test_extract_occurances(mock_species_shapefile_path, mock_processed_raster_path):

    occ_dst_builder = OccurrancesDatasetBuilder(
        [mock_processed_raster_path, mock_processed_raster_path]
    )
    df = occ_dst_builder.build(
        ShapefileLoader(mock_species_shapefile_path).load_dataset()
    )
    assert df.empty is False
