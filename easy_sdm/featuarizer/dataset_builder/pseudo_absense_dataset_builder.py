from pathlib import Path

import pandas as pd
from easy_sdm.utils.data_loader import RasterLoader
import rasterio
from easy_sdm.configs import configs
from easy_sdm.enums import PseudoSpeciesGeneratorType
from .pseudo_species_generators import RSEPPseudoSpeciesGenerator
from easy_sdm.environment import EnverionmentLayersStacker


class PseudoAbsensesDatasetBuilder:
    def __init__(
        self,
        ps_generator_type: PseudoSpeciesGeneratorType,
        region_mask_raster_path: Path,
        stacked_raster_coverages_path: Path,
    ):
        self.ps_generator_type = ps_generator_type
        self.region_mask_raster_path = region_mask_raster_path
        self.stacked_raster_coverages_path = stacked_raster_coverages_path

        self.__define_ps_generator()

    def __get_var_names_list(self):
        return [Path(path).name.split(".")[0] for path in self.raster_path_list]

    def __define_ps_generator(self):
        region_mask_raster = RasterLoader(self.region_mask_raster_path).load_dataset()

        if self.ps_generator_type is PseudoSpeciesGeneratorType.RSEP:
            hyperparameters = configs["pseudo_species"]["RSEP"]
            stacked_raster_coverages = EnverionmentLayersStacker().load(
                self.stacked_raster_coverages_path
            )
            ps_generator = RSEPPseudoSpeciesGenerator(
                hyperparameters=hyperparameters,
                region_mask_raster=region_mask_raster,
                stacked_raster_coverages=stacked_raster_coverages,
            )
        elif self.ps_generator_type is PseudoSpeciesGeneratorType.Random:
            raise NotImplementedError()

        self.ps_generator = ps_generator

    def build(self, occurrence_df: pd.DataFrame, number_pseudo_absenses: int):
        self.ps_generator.fit(occurrence_df)
        pseudo_absenses_df = self.ps_generator.generate(number_pseudo_absenses)
        pseudo_absenses_df["label"] = 0
        return pseudo_absenses_df
