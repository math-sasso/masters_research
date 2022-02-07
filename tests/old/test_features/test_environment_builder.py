from pathlib import Path


from pathlib import Path

from easy_sdm.featuarizer import EnverionmentLayersStacker


# def test_build_environment(processed_raster_paths_list: Path):
#     num_vars = 5
#     processed_raster_paths_list = processed_raster_paths_list[:num_vars]
#     stack = EnverionmentLayersStacker().stack(processed_raster_paths_list)
#     assert stack.shape[0] == num_vars

def test_build_environment2(processed_raster_paths_list: Path):
    env_layer_stacker = EnverionmentLayersStacker()
    output_path = Path.cwd()/ 'data/numpy/env_stack.npy'
    env_layer_stacker.stack_and_save(raster_path_list= processed_raster_paths_list,output_path = output_path)
    import pdb;pdb.set_trace()
    env_layer_stacker.load(output_path)