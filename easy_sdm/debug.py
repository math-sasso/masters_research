from data.shapefile import SpeciesInShapefileChecker
from data.gbif import Species
from pathlib import Path

mays_code = 5290052
mays = Species(taxon_key=5290052, species_name="Zea mays")
mays_gdf = mays.get_species_gdf()
brazil_shapefile_path = Path.cwd() / "data/raw/shapefiles_brasil/level_0/BRA_adm0.shp"
shp_region = SpeciesInShapefileChecker(brazil_shapefile_path)
teste_gdf = shp_region.get_points_inside(mays_gdf)
import pdb

pdb.set_trace()
