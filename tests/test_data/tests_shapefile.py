def test_point_insede_polygon():
    from easy_sdm.data.shapefile import ShapefileRegion
    from easy_sdm.data.gbif import Species, SpeciesGDFBuilder
    from pathlib import Path
    import geopandas as gpd
    mays_code = 5290052
    mays = Species(taxon_key=mays_code,species_name="Zea mays")
    mays_gdf = mays.get_species_gdf()
    brazil_shapefile_path = Path.cwd() / 'data/raw/shapefiles_brasil/level_0/BRA_adm0.shp'
    shp_region = ShapefileRegion(brazil_shapefile_path)
    new_mays_gdf = shp_region.get_points_inside(mays_gdf)
    assert type(new_mays_gdf) is gpd.GeoDataFrame
    assert len(new_mays_gdf) <= len(mays_gdf)
