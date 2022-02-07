from pathlib import Path

import pandas as pd
import rasterio
from easy_sdm.configs import configs
from easy_sdm.enums import PseudoSpeciesGeneratorType
from .pseudo_species_generators import RSEPPseudoSpeciesGenerator
from ..environment_builder.environment_layer_stacker import EnverionmentLayersStacker


class PseudoAbsensesDatasetBuilder:
    def __init__(
        self, ps_generator_type: PseudoSpeciesGeneratorType,
    ):
        self.ps_generator_type = ps_generator_type

        self.__define_ps_generator()

    def __define_ps_generator(self):

        region_mask_raster = rasterio.open(
            Path.cwd() / "data/processed_rasters/others/brazilian_mask.tif"
        )

        if self.ps_generator_type is PseudoSpeciesGeneratorType.RSEP:
            hyperparameters = configs["pseudo_species"]["RSEP"]
            stacked_raster_coverages = EnverionmentLayersStacker().load(
                Path.cwd() / "data/numpy/env_stack.npy"
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
        return pseudo_absenses_df
