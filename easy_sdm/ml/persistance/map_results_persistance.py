from abc import abstractmethod
from cProfile import label
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typos import Species

from easy_sdm.raster_processing.processing.raster_information_extractor import (
    RasterInfoExtractor,
)
from easy_sdm.utils import DatasetLoader, RasterLoader
from easy_sdm.utils.path_utils import PathUtils


class MapResultsPersistance:
    def __init__(self, species: Species, data_dirpath: Path) -> None:
        self.species = species
        self.data_dirpath = data_dirpath
        self.info_extractor = RasterInfoExtractor(
            RasterLoader(
                data_dirpath / "raster_processing/region_mask.tif"
            ).load_dataset()
        )
        self.__setup_vectors()
        self.__setup_colormap()

    def __setup_vectors(self):

        xgrid = self.info_extractor.get_xgrid()
        ygrid = self.info_extractor.get_ygrid()

        X, Y = xgrid, ygrid[::-1]

        self.X = X
        self.Y = Y
        self.land_reference_array = self.info_extractor.get_array()

    def __setup_colormap(self):
        norm = matplotlib.colors.Normalize(-0.001, 1)
        colors = [
            [norm(-0.001), "white"],
            [norm(0.15), "0.95"],
            [norm(0.2), "sienna"],
            [norm(0.3), "wheat"],
            [norm(0.5), "cornsilk"],
            [norm(0.95), "yellowgreen"],
            [norm(1.0), "green"],
        ]

        custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        custom_cmap.set_bad(color="white")
        self.custom_cmap = custom_cmap

    @abstractmethod
    def plot_map(self, Z: np.ndarray, estimator_type_text: str):
        pass


class MapResultsPersistanceWithoutCoords(MapResultsPersistance):
    def __init__(self, species: Species, data_dirpath: Path) -> None:
        super().__init__(species, data_dirpath)

    def plot_map(
        self,
        Z: np.ndarray,
        estimator_type_text: str,
        vif_columns_identifier: str,
        experiment_dirpath: str,
    ):

        plt.figure(figsize=(8, 8))

        # Setting titles and labels
        plt.title(
            f"Distribuição predita para a \nespécie {self.species.get_name_for_plots()}\n algoritmo {estimator_type_text}",
            fontsize=20,
        )

        plt.ylabel("Latitude[graus]", fontsize=18)
        plt.xlabel("Longitude[graus]", fontsize=18)

        # Plot country map
        plt.contour(
            self.X,
            self.Y,
            self.land_reference_array,
            levels=[10],
            colors="k",
            linestyles="solid",
        )

        # print('levels: ',levels)
        plt.contourf(self.X, self.Y, Z, levels=10, cmap=self.custom_cmap)
        plt.colorbar(format="%.2f")

        # Saving results
        plt.legend(loc="upper right")
        output_dirpath = (
            self.data_dirpath
            / f"visualization/{self.species.get_name_for_paths()}/{estimator_type_text}/{vif_columns_identifier}"
        )

        PathUtils.create_folder(output_dirpath)
        output_path = output_dirpath / f"map_without_coords.png"
        plt.savefig(output_path)
        plt.clf()
        return output_path


class MapResultsPersistanceWithCoords(MapResultsPersistance):
    def __init__(self, species: Species, data_dirpath: Path) -> None:
        super().__init__(species, data_dirpath)

    def _extract_occurrence_coords(self, experiment_dirpath):
        coords_df, _ = DatasetLoader(
            experiment_dirpath / "coords_df.csv"
        ).load_dataset()
        sdm_df, _ = DatasetLoader(experiment_dirpath / "complete_df.csv").load_dataset()
        df_occ = sdm_df.loc[sdm_df["label"] == 1]
        coords_occ_df = coords_df.iloc[list(df_occ.index)]
        coords = coords_occ_df.to_numpy()
        return coords

    def _extract_pseudo_absense_coords(self, experiment_dirpath):
        coords_df, _ = DatasetLoader(
            experiment_dirpath / "coords_df.csv"
        ).load_dataset()
        sdm_df, _ = DatasetLoader(experiment_dirpath / "complete_df.csv").load_dataset()
        df_psa = sdm_df.loc[sdm_df["label"] == 0]
        coords_psa_df = coords_df.iloc[list(df_psa.index)]
        coords = coords_psa_df.to_numpy()
        return coords

    def plot_map(
        self,
        Z: np.ndarray,
        estimator_type_text: str,
        vif_columns_identifier: str,
        experiment_dirpath: str,
    ):

        output_dirpath = (
            self.data_dirpath
            / f"visualization/{self.species.get_name_for_paths()}/{estimator_type_text}/{vif_columns_identifier}"
        )

        PathUtils.create_folder(output_dirpath)
        output_path = output_dirpath / f"map_with_coords.png"

        plt.figure(figsize=(8, 8))

        # Setting titles and labels
        plt.title(
            f"Distribuição predita para a \nespécie {self.species.get_name_for_plots()}\n algoritmo {estimator_type_text}",
            fontsize=20,
        )

        plt.ylabel("Latitude [graus]", fontsize=18)
        plt.xlabel("Longitude [graus]", fontsize=18)

        # Plot country map
        plt.contour(
            self.X,
            self.Y,
            self.land_reference_array,
            levels=[10],
            colors="k",
            linestyles="solid",
        )

        # print('levels: ',levels)
        plt.contourf(self.X, self.Y, Z, levels=10, cmap=self.custom_cmap)
        plt.colorbar(format="%.2f")

        occ_coords = self._extract_occurrence_coords(experiment_dirpath)
        psa_coords = self._extract_pseudo_absense_coords(experiment_dirpath)

        plt.scatter(
            occ_coords[:, 1],
            occ_coords[:, 0],
            s=2 ** 3,
            c="blue",
            marker="^",
            label="Occurrance coordinates",
        )
        plt.scatter(
            psa_coords[:, 1],
            psa_coords[:, 0],
            s=2 ** 3,
            c="red",
            marker="^",
            label="Pseudo absenses coordinates",
        )

        # Saving results
        plt.legend(loc="upper right")
        plt.savefig(output_path)
        plt.clf()
        return output_path
