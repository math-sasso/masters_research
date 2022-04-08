from distutils.command.config import config
from gc import callbacks
import pandas as pd
from pathlib import Path
from typing import Optional

from setuptools import setup
import xgboost
from easy_sdm.enums.estimator_type import EstimatorType

import typer

from easy_sdm.download import DownloadJob
from easy_sdm.enums import PseudoSpeciesGeneratorType, ModellingType
from easy_sdm.environment import EnvironmentCreationJob, RelevantRastersSelector
from easy_sdm.featuarizer import DatasetCreationJob
from easy_sdm.raster_processing import RasterProcessingJob
from easy_sdm.species_collection import SpeciesCollectionJob
from easy_sdm.utils import PathUtils
from easy_sdm.utils.data_loader import DatasetLoader, ShapefileLoader
from easy_sdm.ml import TrainJob
from easy_sdm.typos import Species
app = typer.Typer()

milpa_species_dict = {
    5290052:"Zea mays",
    7393329:"Cucurbita moschata",
    2874515:"Cucurbita maxima",
    2874508:"Cucurbita pepo",
    5350452:"Phaseolus vulgaris",
    2982583:"Vigna unguiculata",
    7587087:"Cajanus cajan",
    3086357:"Piper nigrum",
    2932944:"Capsicum annuum",
    2932938:"Capsicum baccatum",
    8403992:"Capsicum frutescens",
    2932942:"Capsicum chinense",
}

def version_callback(value: bool):
    if value:
        with open(Path(__file__).parent / "VERSION", mode="r") as file:
            version = file.read().replace("\n", "")
        typer.echo(f"{version}")
        raise typer.Exit()


@app.callback("version")
def version(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback, is_eager=True,
    ),
):
    """
    Any issue please contact authors
    """
    typer.echo("easy_sdm")


@app.command("download-data")
def download_data():
    raw_rasters_dirpath = Path.cwd() / "data/download/raw_rasters"
    download_job = DownloadJob(raw_rasters_dirpath=raw_rasters_dirpath)
    # download_job.download_shapefile_region()
    download_job.download_soigrids_rasters(coverage_filter="mean")
    # download_job.download_bioclim_rasters()
    # download_job.download_envirem_rasters()


@app.command("process-rasters")
def process_rasters():

    region_shapefile_path = Path.cwd() / "data/download/region_shapefile"
    processed_rasters_dir = Path.cwd() / "data/raster_processing"
    raw_rasters_dir = Path.cwd() / "data/download/raw_rasters"

    raster_processing_job = RasterProcessingJob(
        processed_rasters_dir=processed_rasters_dir, raw_rasters_dir=raw_rasters_dir
    )

    raster_processing_job.process_rasters_from_all_sources()
    raster_processing_job.build_mask(region_shapefile_path)


@app.command("build-species-data")
def build_species_data(
    species_id: int = typer.Option(..., "--species-id", "-s"),
):
    output_dirpath = Path.cwd() / "data/species_collection"
    region_shapefile_path = Path.cwd() / "data/download/region_shapefile"
    species = Species(taxon_key=species_id,name=milpa_species_dict[species_id])
    job = SpeciesCollectionJob(
        output_dirpath=output_dirpath, region_shapefile_path=region_shapefile_path
    )
    job.collect_species_data(
        species=species
    )


@app.command("create-environment")
def create_environment():

    processed_rasters_dir = (
        Path.cwd() / "data/raster_processing/environment_variables_rasters"
    )

    # tomar muito cuidado com essa lista porque a ordem fica baguncada
    # o stacker tem que manter a mesma ordem dos dataframes
    all_rasters_path_list = PathUtils.get_rasters_filepaths_in_dir(
        processed_rasters_dir
    )

    output_dirpath = Path.cwd() / "data/environment"
    env_creation_job = EnvironmentCreationJob(
        output_dirpath=output_dirpath, all_rasters_path_list=all_rasters_path_list
    )
    env_creation_job.build_environment()

def create_dataset_by_specie(
    species_id:int,
    ps_generator_type:str=None,
    ps_proportion:float=None,
):
    species = Species(taxon_key=species_id,name=milpa_species_dict[species_id])


    raster_path_list_path = Path.cwd() / "data/environment/relevant_raster_list"

    raster_path_list = RelevantRastersSelector().load_raster_list(
        raster_list_path=raster_path_list_path
    )

    ps_generator_type = {
        "RSEP": PseudoSpeciesGeneratorType.RSEP,
        "Random": PseudoSpeciesGeneratorType.Random,
    }.get(ps_generator_type, f"{ps_generator_type}' is not supported!")

    featuarizer_dirpath = Path.cwd() / "data/featuarizer"
    stacked_raster_coverages_path = (
        Path.cwd() / "data/environment/environment_stack.npy"
    )
    region_mask_raster_path = Path.cwd() / "data/raster_processing/region_mask.tif"

    sdm_dataset_creator = DatasetCreationJob(
        raster_path_list=raster_path_list,
        ps_generator_type=ps_generator_type,
        ps_proportion=ps_proportion,
        featuarizer_dirpath=featuarizer_dirpath,
        region_mask_raster_path=region_mask_raster_path,
        stacked_raster_coverages_path=stacked_raster_coverages_path,
    )

    species_gdf = ShapefileLoader(
        shapefile_path=Path.cwd() / Path("data/species_collection") / species.get_name_for_paths()
    ).load_dataset()

    df_sdm_bc = sdm_dataset_creator.create_binary_classification_dataset( species_gdf=species_gdf)
    sdm_dataset_creator.save_dataset(species=species,df=df_sdm_bc,modellting_type=ModellingType.BinaryClassification)

@app.command("create-dataset")
def create_dataset(
    species_id: int = typer.Option(..., "--species-id", "-s"),
    ps_generator_type: str = typer.Option(..., "--ps-generator-type", "-t"),
    ps_proportion: float = typer.Option(..., "--ps-proportion", "-p"),
):

    create_dataset_by_specie(
        species_id=species_id,
        ps_generator_type=ps_generator_type,
        ps_proportion=ps_proportion
    )

@app.command("create-all-species-datasets")
def create_dataset(
    ps_generator_type: str = typer.Option(..., "--ps-generator-type", "-t"),
    ps_proportion: float = typer.Option(..., "--ps-proportion", "-p"),
):
    for species_id,_ in milpa_species_dict:
        create_dataset_by_specie(
            species_id=species_id,
            ps_generator_type=ps_generator_type,
            ps_proportion=ps_proportion
        )

@app.command("train")
def train(
    species_id: int = typer.Option(..., "--species-id", "-s"),
    estimator_type : str =  typer.Option(..., "--estimator", "-e"),
):
    # estimator selection
    estimator_type = {
        "mlp": EstimatorType.MLP,
        "gradient_boosting": EstimatorType.GradientBoosting,
        "ensemble_forest": EstimatorType.EnsembleForest,
        "xgboost": EstimatorType.Xgboost,
        "xgboostrf": EstimatorType.XgboostRF,
        "tabnet": EstimatorType.Tabnet,
        "ocsvm": EstimatorType.OCSVM,
        "autoencoder": EstimatorType.Autoencoder,

    }.get(estimator_type, f"{estimator_type}' is not supported!")

    # modellling Type slection
    modelling_type = {
        EstimatorType.MLP : ModellingType.BinaryClassification,
        EstimatorType.GradientBoosting : ModellingType.BinaryClassification,
        EstimatorType.EnsembleForest : ModellingType.BinaryClassification,
        EstimatorType.Xgboost : ModellingType.BinaryClassification,
        EstimatorType.XgboostRF : ModellingType.BinaryClassification,
        EstimatorType.Tabnet : ModellingType.BinaryClassification,
        EstimatorType.OCSVM : ModellingType.AnomalyDetection,
        EstimatorType.Autoencoder : ModellingType.AnomalyDetection,
    }.get(estimator_type, None)

    # useful info
    species = Species(taxon_key=species_id,name=milpa_species_dict[species_id])
    datasets_dirpath = Path.cwd() / f"data/featuarizer/datasets/{species.get_name_for_paths()}/{modelling_type.value}"

    # dataloaders
    train_data_loader = DatasetLoader(dataset_path=datasets_dirpath / "train.csv", output_column='label')
    validation_data_loader = DatasetLoader(dataset_path=datasets_dirpath / "valid.csv", output_column='label')
    vif_train_data_loader = DatasetLoader(dataset_path=datasets_dirpath / "vif_train.csv", output_column='label')
    vif_validation_data_loader = DatasetLoader(dataset_path=datasets_dirpath / "vif_valid.csv", output_column='label')

    # train job
    train_job = TrainJob(
        train_data_loader = train_data_loader,
        validation_data_loader = validation_data_loader,
        estimator_type = estimator_type,
        species=species,
    )

    train_job.fit()
    train_job.persist()

    train_job.vif_setup(
        vif_train_data_loader=vif_train_data_loader,
        vif_validation_data_loader=vif_validation_data_loader
    )

    train_job.fit()
    train_job.persist()

# @app.command("download_raw_data")
# Ver o diagrama, essa parte vai ter que passsar pelo seleto
# @app.command("train")
# @app.command("visualize")


# # poetry run python3 easy_sdm/console.py download-soigrids-all-rasters -f mean
# @app.command("download-soigrids-all-rasters")
# def download_soigrids_all(
#     coverage_filter: str = typer.Option(..., "--coverage-filter", "-f"),
# ):

#     download_soigrids_all_rasters(coverage_filter=coverage_filter,)


# @app.command("download-soilgrids-one-rasters")
# def download_soilgrids_one(
#     variable: str = typer.Option(..., "--variable", "-v"),
#     coverage_filter: str = typer.Option(..., "--coverage-filter", "-f"),
# ):

#     download_soilgrods_one_raster(
#         variable=variable, coverage_filter=coverage_filter,
#     )


# def raster_type_callback(value: str):
#     if value not in ["soilgrids", "bioclim", "elevation", "envirem"]:
#         typer.echo(
#             "raster-type must be one between ['soilgrids','bioclim','elevation','envirem']"
#         )
#         raise typer.Exit()
#     return value


# # poetry run python3 easy_sdm/console.py standarize-raster -s data/raw/rasters/Soilgrids_Rasters -d data/processed_rasters/standarized_rasters/Soilgrids_Rasters -t soilgrids
# @app.command("standarize-raster")
# def standarize_rasters_console(
#     source_dirpath: Path = typer.Option(..., "--source-dirpath", "-s"),
#     destination_dirpath: Path = typer.Option(..., "--destination-dirpath", "-d"),
#     raster_type: str = typer.Option(
#         ..., "--raster-type", "-t", callback=raster_type_callback
#     ),
# ):

#     standarize_rasters(
#         source_dirpath=source_dirpath,
#         destination_dirpath=destination_dirpath,
#         raster_type=raster_type,
#     )


# @app.command("save-specie")
# def save_specie(
#     species_id: int = typer.Option(..., "--species-id", "-i"),
#     species_name: str = typer.Option(..., "--species-name", "-n"),
#     shapefile_region_delimiter_path: Path = typer.Option(
#         ..., "--shapefile-region_delimiter-path", "-s"
#     ),
# ):
#     destination_dirpath = Path.cwd() / "data/species_data/occurances"
#     shp_region = SpeciesInShapefileChecker(Path.cwd() / shapefile_region_delimiter_path)
#     collect_species_data(
#         species_id=species_id,
#         species_name=species_name,
#         shp_region=shp_region,
#         output_dir=destination_dirpath,
#     )


# @app.command("save-milpa-species")
# def save_milpa_species(
#     shapefile_region_delimiter_path: Path = typer.Option(
#         ..., "--shapefile-region_delimiter-path", "-s"
#     ),
# ):
#     species_dir = {
#         "Zea mysa": 5290052,
#         "Cucurbita moschata": 7393329,
#         "Cucurbita maxima": 2874515,
#         "Cucurbita pepo": 2874508,
#         "Phaseolus vulgaris": 5350452,
#         "Vigna unguiculata": 2982583,
#         "Cajanus cajan": 7587087,
#         "Piper nigrum": 3086357,
#         "Capsicum annuum": 2932944,
#         "Capsicum baccatum": 2932938,
#         "Capsicum frutescens": 8403992,
#         "Capsicum chinense": 2932942,
#     }

#     rare_species_dir = {
#         "Zea diploperennis": 5290048,
#         "Zea luxurians": 5290053,
#         "Zea nicaraguensis": 5678409,
#         "Zea perennis": 5290054,
#         "Cucurbita ficifolia": 2874512,
#         "Cucurbita argyrosperma": 7907172,
#         "Capsicum pubescens": 2932943,
#     }

#     shp_region = SpeciesInShapefileChecker(Path.cwd() / shapefile_region_delimiter_path)
#     destination_dirpath = Path.cwd() / "data/species_data/occurances"
#     for species_name, species_id in species_dir.items():

#         collect_species_data(
#             species_id=species_id,
#             species_name=species_name,
#             shp_region=shp_region,
#             output_dir=destination_dirpath,
#         )


if __name__ == "__main__":
    app()
