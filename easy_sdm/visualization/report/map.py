import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from easy_sdm.raster_processing.processing.raster_information_extractor import (
    RasterInfoExtractor,
)
from typos import Species
import matplotlib.colors as mcolors
from easy_sdm.utils import NumpyArrayLoader


class MapPlotter(object):
    def __init__(self, species: Species) -> None:
        self.species = species
        raster_processing_dirpath = Path.cwd() / "data/raster_processing"
        self.country_mask_path = RasterInfoExtractor(
            raster_processing_dirpath / "region_mask.tif"
        )
        self.info_extractor = RasterInfoExtractor(self.country_mask_path)
        self.enviroment_dirpath = Path.cwd() / "data/environment/environment_stack.npy"
        self.enviroment_dirpath = (
            Path.cwd() / "data/featuarizer/datasets/zea_mays/binary_classification"
        )

    def create_Z(self):

        # stacked_raster_coverages = utils_methods.retrieve_data_from_np_array(stacked_environment_rasters_array_path)
        # idx = np.where(land_reference == raster_utils.positive_mask_val) # Coords X and Y in two tuples where condition matchs (array(),array())
        # brazil_vars_mean_std_df = pd.read_csv(mean_std_path)
        # mean_vars = np.float32(brazil_vars_mean_std_df['mean'].to_numpy())
        # std_vars = np.float32(brazil_vars_mean_std_df['std'].to_numpy())

        # stacked_raster_coverages_shape = stacked_raster_coverages.shape
        # raster_coverages_land = stacked_raster_coverages[:, idx[0], idx[1]].T
        # for k in range(raster_coverages_land.shape[1]):
        # raster_coverages_land[:,k][raster_coverages_land[:,k]<= raster_utils.no_data_val] = mean_vars[k]
        # del stacked_raster_coverages

        # scaled_coverages_land = (raster_coverages_land - mean_vars) / std_vars
        # scaled_coverages_land[np.isnan(scaled_coverages_land)] = 0
        # del raster_coverages_land

        # global_pred = loaded_model.predict(x=scaled_coverages_land) ###predict output value (N,38)

        # Z= np.ones((stacked_raster_coverages_shape[1], stacked_raster_coverages_shape[2]), dtype=np.float32)
        # # Z *= global_pred.min()
        # # Z *=-1 #This will be necessary to set points outside map to the minimum
        # Z*= self.configs["maps"]["no_data_val"] #This will be necessary to set points outside map to the minimum
        # Z[idx[0], idx[1]] = global_pred.ravel()
        # Z[Z == raster_utils.no_data_val] = -0.001

        # return Z
        pass

    def setup(self):

        xgrid = self.info_extractor.get_xgrid()
        ygrid = self.info_extractor.get_ygrid()
        land_reference_array = self.info_extractor.get_array()

        # Getting Meshgrids
        # X, Y = np.meshgrid(xgrid, ygrid[::-1])

        X, Y = xgrid, ygrid[::-1]

        plt.figure(figsize=(8, 8))
        Z = NumpyArrayLoader()
        Z = utils_methods.retrieve_data_from_np_array(
            os.path.join(fold, "Land_Prediction.npy")
        )

    def create_result_adaptabilities_map(
        self, output_folder, used_algorithim, n_levels,
    ):

        plt.figure(figsize=(8, 8))

        # Setting titles and labels
        plt.title(f"Distribuição predita para a \nespécie {species_name}", fontsize=20)
        plt.ylabel("Latitude[graus]", fontsize=18)
        plt.xlabel("Longitude[graus]", fontsize=18)

        # Plot country map
        plt.contour(X, Y, land_reference, levels=[10], colors="k", linestyles="solid")

        # print('levels: ',levels)
        plt.contourf(X, Y, Z, levels=10, cmap=custom_cmap)
        plt.colorbar(format="%.2f")

        # Saving results
        plt.legend(loc="upper right")
        output_folder = os.path.join(results_folder, species_name)
        output_folder = os.path.join(output_folder, used_algorithim)
        utils_methods.create_folder_structure(output_folder)
        plt.savefig(
            f"{output_folder}/land_map_final_prediction_{used_algorithim}_strategy.png"
        )
        plt.show()
        plt.clf()

    def create_result_adaptabilities_map_wtih_coords(self):
        import pdb

        pdb.set_trace()
