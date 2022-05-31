from pathlib import Path


class EDAJob:
    def __init__(self, data_dirpath: Path) -> None:
        self.data_dirpath = data_dirpath

    def count_occurances_per_species(self):
        species_dirpath = self.data_dirpath / "species_collection"
        for path in species_dirpath.glob("**/*.shp"):
            ...
