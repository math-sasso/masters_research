from pathlib import Path
from pathlib import Path
from typing import Optional
from pathlib import Path

import typer

from preprocessing import standarize_rasters

app = typer.Typer()


@app.command("standarize-raster")
def standarize_rasters_console(
    source_dirpath: Path = typer.Option(..., "--source-dirpath", "-s"),
    destination_dirpath: Path = typer.Option(..., "--destination-dirpath", "-d"),
):
    standarize_rasters(
        source_dirpath=source_dirpath,
        destination_dirpath=destination_dirpath,
    )


if __name__ == "__main__":
    app()
