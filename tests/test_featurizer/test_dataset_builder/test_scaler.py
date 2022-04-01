from pathlib import Path


from pathlib import Path

from easy_sdm.featuarizer import MinMaxScalerWrapper
from easy_sdm.environment import EnverionmentLayersStacker

def test_scale_enverioment(mock_processed_raster_path):
    processed_raster_paths_list = [mock_processed_raster_path,mock_processed_raster_path]
    num_vars = len(processed_raster_paths_list)
    processed_raster_paths_list = processed_raster_paths_list[:num_vars]
    stack = EnverionmentLayersStacker(processed_raster_paths_list).stack()
    scaler_wraper = MinMaxScalerWrapper(raster_path_list=processed_raster_paths_list)
    scaled_stack = scaler_wraper.scale_stack(stack=stack, statistics_dataset=df_stats)
    import pdb;pdb.set_trace()
    assert scaled_stack.shape[0] == num_vars

def test_scale_dataset(processed_raster_paths_list: Path):
    pass
    # num_vars = 5
    # processed_raster_paths_list = processed_raster_paths_list[:num_vars]
    # MinMaxScalerWrapper(raster_path_list=processed_raster_paths_list)
