from pathlib import Path
from easy_sdm.utils import ShapefileLoader


class EDAJob:
    def __init__(self, data_dirpath: Path) -> None:
        self.data_dirpath = data_dirpath

    def count_occurances_per_species(self):
        dict_species = {}
        species_dirpath = self.data_dirpath / "species_collection"
        for path in species_dirpath.glob("**/*.shp"):
            species_name = path.name.replace(".shp", "")
            species_gdf = ShapefileLoader(path).load_dataset()
            dict_species[species_name] = len(species_gdf)

        return dict_species
