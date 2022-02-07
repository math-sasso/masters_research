from pathlib import Path
from .species_data.species_in_shapefile_checker import SpeciesInShapefileChecker
from .species_data.species import Species
from .species_data.species_gdf_builder import SpeciesGDFBuilder


class SpeciesProcessingJob:
    def __init__(self) -> None:
        self.shp_region = SpeciesInShapefileChecker(
            Path.cwd() / "data/raw/shapefiles_brasil/level_0/BRA_adm0.shp"
        )

    def collect_species_data(
        species_id: int,
        species_name: str,
        shp_region: SpeciesInShapefileChecker,
        output_dir: Path,
    ):
        species_gdf_builder = SpeciesGDFBuilder(
            Species(taxon_key=species_id, name=species_name), shp_region
        )
        species_name = species_name.replace(" ", "_")
        path = output_dir / species_name / f"{species_name}.shp"
        species_gdf_builder.save_species_gdf(path)

    shp_region = SpeciesInShapefileChecker(Path.cwd() / shapefile_region_delimiter_path)
    destination_dirpath = Path.cwd() / "data/species_data/occurances"
    for species_name, species_id in species_dir.items():

        collect_species_data(
            species_id=species_id,
            species_name=species_name,
            shp_region=shp_region,
            output_dir=destination_dirpath,
        )
