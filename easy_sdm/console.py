from distutils.command.config import config
from gc import callbacks
import pandas as pd
from pathlib import Path
from typing import Optional

from setuptools import setup
from easy_sdm.enums.estimator_type import EstimatorType
from easy_sdm.enums.raster_source import RasterSource
from easy_sdm.ml.prediction_job import Prediction_Job

import typer

from easy_sdm.download import DownloadJob
from easy_sdm.enums import PseudoSpeciesGeneratorType, ModellingType
from easy_sdm.environment import EnvironmentCreationJob
from easy_sdm.featuarizer import DatasetCreationJob
from easy_sdm.raster_processing import RasterProcessingJob
from easy_sdm.species_collection import SpeciesCollectionJob
from easy_sdm.utils import PathUtils
from easy_sdm.utils.data_loader import DatasetLoader, ShapefileLoader, PickleLoader
from easy_sdm.ml import TrainJob
from easy_sdm.typos import Species

app = typer.Typer()

milpa_species_dict = {
    5290052: "Zea mays",
    7393329: "Cucurbita moschata",
    2874515: "Cucurbita maxima",
    2874508: "Cucurbita pepo",
    5350452: "Phaseolus vulgaris",
    2982583: "Vigna unguiculata",
    7587087: "Cajanus cajan",
    3086357: "Piper nigrum",
    2932944: "Capsicum annuum",
    2932938: "Capsicum baccatum",
    8403992: "Capsicum frutescens",
    2932942: "Capsicum chinense",
}

data_dirpath = Path.cwd() / "data"
all_algorithims_string = "mlp, gradient_boosting, ensemble_forest, xgboost, xgboostrf, tabnet, ocsvm, autoencoder"
# estimator selection
def estimator_type_selector(estimator_type: str):
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
    return estimator_type


# modellling Type slection from estimator
def modellling_type_selector_from_estimator(estimator_type: str):
    modelling_type = {
        EstimatorType.MLP: ModellingType.BinaryClassification,
        EstimatorType.GradientBoosting: ModellingType.BinaryClassification,
        EstimatorType.EnsembleForest: ModellingType.BinaryClassification,
        EstimatorType.Xgboost: ModellingType.BinaryClassification,
        EstimatorType.XgboostRF: ModellingType.BinaryClassification,
        EstimatorType.Tabnet: ModellingType.BinaryClassification,
        EstimatorType.OCSVM: ModellingType.AnomalyDetection,
        EstimatorType.Autoencoder: ModellingType.AnomalyDetection,
    }.get(estimator_type, None)
    return modelling_type


def ps_generator_type_selector(ps_generator_type):
    ps_generator_type = {
        "RSEP": PseudoSpeciesGeneratorType.RSEP,
        "Random": PseudoSpeciesGeneratorType.Random,
    }.get(ps_generator_type, f"{ps_generator_type}' is not supported!")

    return ps_generator_type


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
    raw_rasters_dirpath = data_dirpath / "download/raw_rasters"
    download_job = DownloadJob(raw_rasters_dirpath=raw_rasters_dirpath)
    # download_job.download_shapefile_region()
    download_job.download_soigrids_rasters(coverage_filter="mean")
    # download_job.download_bioclim_rasters()
    # download_job.download_envirem_rasters()


@app.command("process-rasters")
def process_rasters():

    raster_processing_job = RasterProcessingJob(data_dirpath=data_dirpath)

    raster_processing_job.process_rasters_from_all_sources()


@app.command("build-species-data")
def build_species_data(species_id: int = typer.Option(..., "--species-id", "-s"),):
    output_dirpath = data_dirpath / "species_collection"
    region_shapefile_path = data_dirpath / "download/region_shapefile"
    species = Species(taxon_key=species_id, name=milpa_species_dict[species_id])
    job = SpeciesCollectionJob(
        output_dirpath=output_dirpath, region_shapefile_path=region_shapefile_path
    )
    job.collect_species_data(species=species)


@app.command("create-environment")
def create_environment():

    processed_rasters_dir = (
        data_dirpath / "raster_processing/environment_variables_rasters"
    )

    # tomar muito cuidado com essa lista porque a ordem fica baguncada
    # o stacker tem que manter a mesma ordem dos dataframes
    all_rasters_path_list = PathUtils.get_rasters_filepaths_in_dir(
        processed_rasters_dir
    )

    output_dirpath = data_dirpath / "environment"
    env_creation_job = EnvironmentCreationJob(
        output_dirpath=output_dirpath, all_rasters_path_list=all_rasters_path_list
    )
    env_creation_job.build_environment()


def create_dataset_by_specie(
    species_id: int, ps_generator_type: str = None, ps_proportion: float = None,
):
    species = Species(taxon_key=species_id, name=milpa_species_dict[species_id])

    ps_generator_type = ps_generator_type_selector(ps_generator_type)

    sdm_dataset_creator = DatasetCreationJob(
        root_data_dirpath=data_dirpath,
        ps_proportion=ps_proportion,
        ps_generator_type=ps_generator_type,
    )

    species_gdf = ShapefileLoader(
        shapefile_path=data_dirpath
        / "species_collection"
        / species.get_name_for_paths()
    ).load_dataset()

    df_sdm_bc, coords_df = sdm_dataset_creator.create_general_dataset(
        species_gdf=species_gdf,
    )

    sdm_dataset_creator.save_dataset(
        species=species,
        sdm_df=df_sdm_bc,
        coords_df=coords_df,
        modellting_type=ModellingType.AnomalyDetection,
    )

    sdm_dataset_creator.save_dataset(
        species=species,
        sdm_df=df_sdm_bc,
        coords_df=coords_df,
        modellting_type=ModellingType.BinaryClassification,
    )


@app.command("create-dataset")
def create_dataset(
    species_id: int = typer.Option(..., "--species-id", "-s"),
    ps_generator_type: str = typer.Option(..., "--ps-generator-type", "-t"),
    ps_proportion: float = typer.Option(..., "--ps-proportion", "-p"),
):

    create_dataset_by_specie(
        species_id=species_id,
        ps_generator_type=ps_generator_type,
        ps_proportion=ps_proportion,
    )


@app.command("create-all-species-datasets")
def create_dataset(
    ps_generator_type: str = typer.Option(..., "--ps-generator-type", "-t"),
    ps_proportion: float = typer.Option(..., "--ps-proportion", "-p"),
):
    for species_id, _ in milpa_species_dict:
        create_dataset_by_specie(
            species_id=species_id,
            ps_generator_type=ps_generator_type,
            ps_proportion=ps_proportion,
        )


@app.command("train")
def train(
    species_id: int = typer.Option(..., "--species-id", "-s"),
    estimator_type: str = typer.Option(
        ..., "--estimator", "-e", help=f"Must be one between {all_algorithims_string}"
    ),
):

    estimator_type = estimator_type_selector(estimator_type)
    modelling_type = modellling_type_selector_from_estimator(estimator_type)
    # useful info
    species = Species(taxon_key=species_id, name=milpa_species_dict[species_id])
    dataset_dirpath = (
        data_dirpath
        / f"featuarizer/datasets/{species.get_name_for_paths()}/{modelling_type.value}"
    )
    # dataloaders
    train_data_loader = DatasetLoader(
        dataset_path=dataset_dirpath / "train.csv", output_column="label"
    )
    validation_data_loader = DatasetLoader(
        dataset_path=dataset_dirpath / "valid.csv", output_column="label"
    )
    vif_train_data_loader = DatasetLoader(
        dataset_path=dataset_dirpath / "vif_train.csv", output_column="label"
    )
    vif_validation_data_loader = DatasetLoader(
        dataset_path=dataset_dirpath / "vif_valid.csv", output_column="label"
    )

    # train job
    train_job = TrainJob(
        train_data_loader=train_data_loader,
        validation_data_loader=validation_data_loader,
        estimator_type=estimator_type,
        species=species,
    )

    train_job.fit()
    train_job.persist()

    train_job.vif_setup(
        vif_train_data_loader=vif_train_data_loader,
        vif_validation_data_loader=vif_validation_data_loader,
    )

    train_job.fit()
    train_job.persist()


@app.command("infer-map")
def infer_map(
    species_id: int = typer.Option(..., "--species-id", "-s"),
    run_id: str = typer.Option(..., "--run_id", "-r"),
):
    # TODO: selecionar o numero de features do vif. Vai precisar saber qual o numero da coluna que vai precisar filtrar
    species = Species(taxon_key=species_id, name=milpa_species_dict[species_id])
    prediction_job = Prediction_Job(data_dirpath=data_dirpath)
    prediction_job.set_model(run_id=run_id, species=species)
    Z = prediction_job.map_prediction()
    prediction_job.log_map(Z=Z)


@app.command("generate-results")
def generate_results(
    species_id: int = typer.Option(..., "--species-id", "-s"),
    map_generation: str = typer.Option(..., "--map-generation", "-mg"),
    model_comparison: bool = typer.Option(..., "--model-comparison", "-mc"),
    estimator_type: str = typer.Option(..., "--estimator", "-e"),
):
    pass


if __name__ == "__main__":
    app()
