from pathlib import Path
from typing import Optional

import typer

from easy_sdm.data.shapefile_data import ShapefileRegion
from easy_sdm.data_colector import standarize_rasters, collect_species_data
from utils.utils import PathUtils

app = typer.Typer()


@app.command("standarize-raster")
def standarize_rasters_console(
    source_dirpath: Path = typer.Option(..., "--source-dirpath", "-s"),
    destination_dirpath: Path = typer.Option(..., "--destination-dirpath", "-d"),
):

    import pdb

    pdb.set_trace()
    standarize_rasters(
        source_dirpath=source_dirpath, destination_dirpath=destination_dirpath,
    )


@app.command("save_specie")
def save_specie(
    species_id: int = typer.Option(..., "--species-id", "-i"),
    species_name: str = typer.Option(..., "--species-name", "-n"),
    shapefile_region_delimiter_path: Path = typer.Option(
        ..., "--shapefile-region_delimiter-path", "-s"
    ),
):
    destination_dirpath = Path.cwd() / "data/species_data/occurances"
    shp_region = ShapefileRegion(Path.cwd() / shapefile_region_delimiter_path)
    collect_species_data(
        species_id=species_id,
        species_name=species_name,
        shp_region=shp_region,
        output_dir=destination_dirpath,
    )


@app.command("save_milpa_species")
def save_milpa_species(
    shapefile_region_delimiter_path: Path = typer.Option(
        ..., "--shapefile-region_delimiter-path", "-s"
    ),
):
    species_dir = {
        "Zea mysa": 5290052,
        "Cucurbita moschata": 7393329,
        "Cucurbita maxima": 2874515,
        "Cucurbita pepo": 2874508,
        "Phaseolus vulgaris": 5350452,
        "Vigna unguiculata": 2982583,
        "Cajanus cajan": 7587087,
        "Piper nigrum": 3086357,
        "Capsicum annuum": 2932944,
        "Capsicum baccatum": 2932938,
        "Capsicum frutescens": 8403992,
        "Capsicum chinense": 2932942,
    }

    rare_species_dir = {
        "Zea diploperennis": 5290048,
        "Zea luxurians": 5290053,
        "Zea nicaraguensis": 5678409,
        "Zea perennis": 5290054,
        "Cucurbita ficifolia": 2874512,
        "Cucurbita argyrosperma": 7907172,
        "Capsicum pubescens": 2932943,
    }

    shp_region = ShapefileRegion(Path.cwd() / shapefile_region_delimiter_path)
    destination_dirpath = Path.cwd() / "data/species_data/occurances"
    for species_name, species_id in species_dir.items():

        collect_species_data(
            species_id=species_id,
            species_name=species_name,
            shp_region=shp_region,
            output_dir=destination_dirpath,
        )


if __name__ == "__main__":
    app()
