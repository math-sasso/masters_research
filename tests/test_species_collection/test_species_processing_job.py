import geopandas as gpd
from easy_sdm.raster_processing import SpeciesCollectionJob
from easy_sdm.utils import ShapefileLoader


def test_get_secies_data(tmp_path, mock_shapefile_region_path):
    species_id = 7587087
    species_name = "Canajus cajan"
    job = SpeciesCollectionJob(
        output_dirpath=tmp_path, region_shapefile_path=mock_shapefile_region_path
    )
    job.collect_species_data(species_id=species_id, species_name=species_name)
    geo_df = ShapefileLoader(tmp_path / species_name.replace(" ", "_")).load_dataset()
    assert "geometry" in geo_df.columns
    assert isinstance(geo_df, gpd.GeoDataFrame)
