class Species:
    def __init__(self, taxon_key: int, name: str):
        self.taxon_key = taxon_key
        self.name = name

    def __str__(self) -> str:
        return "Species {self.name} with taxon key {self.taxon_key}"
