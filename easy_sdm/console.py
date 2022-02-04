from gc import callbacks
from pathlib import Path
from typing import Optional

import typer

from easy_sdm.data import ShapefileRegion
from easy_sdm.data_colector import (
    standarize_rasters,
    collect_species_data,
    download_soilgrods_one_raster,
    download_soigrids_all_rasters,
)

app = typer.Typer()

# poetry run python3 easy_sdm/console.py download-soigrids-all-rasters -f mean
@app.command("download-soigrids-all-rasters")
def download_soigrids_all(
    coverage_filter: str = typer.Option(..., "--coverage-filter", "-f"),
):

    download_soigrids_all_rasters(coverage_filter=coverage_filter,)


@app.command("download-soilgrids-one-rasters")
def download_soilgrids_one(
    variable: str = typer.Option(..., "--variable", "-v"),
    coverage_filter: str = typer.Option(..., "--coverage-filter", "-f"),
):

    download_soilgrods_one_raster(
        variable=variable, coverage_filter=coverage_filter,
    )


def raster_type_callback(value: str):
    if value not in ["soilgrids", "bioclim", "elevation", "envirem"]:
        typer.echo(
            "raster-type must be one between ['soilgrids','bioclim','elevation','envirem']"
        )
        raise typer.Exit()
    return value


# poetry run python3 easy_sdm/console.py standarize-raster -s data/raw/rasters/Soilgrids_Rasters -d data/processed_rasters/standarized_rasters/Soilgrids_Rasters -t soilgrids
@app.command("standarize-raster")
def standarize_rasters_console(
    source_dirpath: Path = typer.Option(..., "--source-dirpath", "-s"),
    destination_dirpath: Path = typer.Option(..., "--destination-dirpath", "-d"),
    raster_type: str = typer.Option(
        ..., "--raster-type", "-t", callback=raster_type_callback
    ),
):

    standarize_rasters(
        source_dirpath=source_dirpath,
        destination_dirpath=destination_dirpath,
        raster_type=raster_type,
    )


@app.command("save-specie")
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


@app.command("save-milpa-species")
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
