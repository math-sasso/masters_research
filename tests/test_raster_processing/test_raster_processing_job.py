def test_raster_processing_job(tmp_path):

    # TODO: Sera necessario refatorar o codigo
    # do jeito que esta ou entao criar um path com a estrutura minima para chamar o job

    # from easy_sdm.data_processing import RasterProcessingJob

    # region_shapefile_path = tmp_path / "data/download/region_shapefile"
    # processed_rasters_dir = tmp_path / "data/data_processing/processed_rasters"
    # raw_rasters_dir = tmp_path / "data/download/raw_rasters"
    # raster_processing_job = RasterProcessingJob(
    #     processed_rasters_dir=processed_rasters_dir, raw_rasters_dir=raw_rasters_dir
    # )

    # raster_processing_job.process_rasters_from_all_sources()
    # raster_processing_job.build_mask(
    #     region_shapefile_path
    # )


# def test_standarize_soilgrids_raster(tmp_path):
#     """[Este teste esta dando errado
#     Uma possivel solucao seria jogar todos os pontos 0.0 para -9999.0 e em seguida filtrar
#     pelo mapa do brasil e jogar todos os que estivessem dentro para 0. Precisa pensar se
#     isso faz sentido.
#     ]
#     """

#     raster_path = "data/raw/rasters/Soilgrids_Rasters/bdod/bdod_15-30cm_mean.tif"
#     output_path = tmp_path / Path(raster_path).name
#     raster_standarizer = RasterStandarizer()
#     raster_standarizer.standarize_soilgrids(
#         input_path=raster_path,
#         output_path=output_path,
#     )
#     standarized_raster = rasterio.open(output_path)

#     assert np.min(standarized_raster.read(1)) is configs["maps"]["no_data_val"]
