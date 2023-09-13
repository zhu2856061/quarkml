# -*- coding: utf-8 -*-
# @Time   : 2023/7/27 09:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function
import pandas as pd
from loguru import logger
from quarkml.generation_operation.baisc_operation import BasicGeneration
from typing import List
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


class FeatureGeneration(object):

    def __init__(self) -> None:
        self.bg = BasicGeneration()

    def basic_operator(
        self,
        X: pd.DataFrame,
        categorical_features: List = None,
        numerical_features: List = None,
        report_dir: str = 'encode',
    ):

        candidate_features = self.bg.fit(
            X,
            categorical_features,
            numerical_features,
            report_dir,
        )
        candidate_features.sort()

        logger.info(f"===== candidate_features number: {len(candidate_features)} =====")
        return candidate_features
