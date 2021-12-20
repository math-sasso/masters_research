from pathlib import Path
from data.data_loader import RasterLoader
from data.raster_standarizer import RasterCliper
from utils.utils import PathUtils


raster_loader = RasterLoader()
raster_cliper = RasterCliper()


def standarize_rasters(source_dirpath: str, destination_dirpath: str):
    # source_dirpath: Path.cwd() / 'data/raw'
    # destination_dirpath: Path.cwd() / 'data/standarized_rasters'
    for name, path in PathUtils.list_rasters_in_folder(source_dirpath):
        try:
            raster = raster_loader.read(path)
            raster_cliper.clip_and_save(
                source_raster=raster, output_path=PathUtils.file_path_existis(destination_dirpath / name)
            )
        except:
            pass
        import pdb
    
        pdb.set_trace()