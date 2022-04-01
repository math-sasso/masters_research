import pytest
from easy_sdm.species_collection import SpeciesCollectionJob
from easy_sdm.configs import configs
from pathlib import Path


@pytest.mark.interface
def test_species_collection_job(tmp_path, mock_map_shapefile_path):
    species_dict = configs["species"]
    species_id = species_dict["id"]
    species_name = species_dict["name"]
    job = SpeciesCollectionJob(
        output_dirpath=tmp_path, region_shapefile_path=mock_map_shapefile_path
    )
    job.collect_species_data(species_id=species_id, species_name=species_name)
