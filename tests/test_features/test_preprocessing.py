from pathlib import Path

import pandas as pd
from easy_sdm.featuarizer import (
    RasterStatisticsCalculator,
)
from easy_sdm.utils import PathUtils


def test_statistics_table_generation(df_stats):
    assert df_stats.empty is False
