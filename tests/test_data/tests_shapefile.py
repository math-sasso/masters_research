def test_species_request(tmp_path,mock_map_shapefile_path):
    from easy_sdm.data import Species, SpeciesGDFBuilder,ShapefileRegion

    import geopandas as gpd
    mays_code = 5290052
    mays = SpeciesGDFBuilder(Species(taxon_key=mays_code,name="Zea mays"))
    mays_gdf = mays.get_species_gdf()
    shp_region = ShapefileRegion(mock_map_shapefile_path)
    new_mays_gdf = shp_region.get_points_inside(mays_gdf)
    mays.save_species_gdf
    assert type(new_mays_gdf) is gpd.GeoDataFrame
    assert len(new_mays_gdf) <= len(mays_gdf)