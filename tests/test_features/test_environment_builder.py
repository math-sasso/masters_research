from pathlib import Path


from pathlib import Path

from easy_sdm.featuarizer import EnverionmentLayersStacker



def test_build_environment(processed_raster_paths_list: Path):
    num_vars = 5
    processed_raster_paths_list = processed_raster_paths_list[:num_vars]
    stack = EnverionmentLayersStacker(processed_raster_paths_list).stack()
    assert stack.shape[0] == num_vars
