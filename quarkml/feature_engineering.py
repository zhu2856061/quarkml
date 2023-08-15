# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

from quarkml.core.data_processing import DataProcessing
from quarkml.core.distributed_data_processing import DistDataProcessing
from quarkml.core.feature_generation import FeatureGeneration
from quarkml.core.feature_selector import FeatureSelector
from typing import List
import pandas as pd

# 特征工程
class FeatureEngineering(object):

    def __init__(self) -> None:
        self.DP = DataProcessing()
        self.DDP = DistDataProcessing()
        self.FG = FeatureGeneration()
        self.FS = FeatureSelector()

    def data_processing(
        self,
        ds: pd.DataFrame,
        label_name: str,
        cat_feature: List = [],
        num_feature: List = [],
        ordinal_number=100,
        report_dir="./encode",
        is_fillna=True,
        drop_outliers=False,
        verbosity=False,
        task='fit',
    ):
        if task == 'fit':
            return self.DP.fit(
                ds,
                label_name,
                cat_feature,
                num_feature,
                ordinal_number,
                report_dir,
                is_fillna,
                drop_outliers,
                verbosity,
            )
        else:
            return self.DP.tranform(
                ds,
                label_name,
                report_dir,
                verbosity,
            )

    def dist_data_processing(
        self,
        files = None,
        label_name: str = None,
        delimiter: str = ',',
        cat_feature: List = [],
        num_feature: List = [],
        ordinal_number=100,
        report_dir="./encode",
        ds: pd.DataFrame = None,
    ):

        if ds is not None:
            return self.DDP.from_pandas(ds, cat_feature)

        return self.DDP.fit(
            files,
            label_name,
            delimiter,
            cat_feature,
            num_feature,
            ordinal_number,
            report_dir,
        )

    def feature_generation(
        self,
        X: pd.DataFrame,
        categorical_features: List = None,
        numerical_features: List = None,
        report_dir: str = 'encode',
        method: str = "basic",
    ):
        if method == "basic":
            return self.FG.basic_operator(
                X,
                categorical_features,
                numerical_features,
                report_dir,
            )

    def feature_selector(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        candidate_features: List[str] = None,
        categorical_features: List[str] = None,
        numerical_features: List[str] = None,
        params: dict = None,
        tmodel_method='mean',
        init_score: pd.DataFrame = None,
        importance_metric: str = 'importance',
        select_method='predictive',
        min_candidate_features=200,
        blocks=2,
        ratio=0.5,
        folds=5,
        seed=2023,
        part_column: str = None,
        part_values: List = None,
        handle_zero="merge",
        bins=10,
        minimum=0.5,
        minimal: int = 1,
        priori=None,
        use_base=True,
        report_dir="encode",
        method: str = "booster",
        distributed_and_multiprocess=-1,
        job=-1,
    ):
        if method == "booster":
            selected_feature, new_X = self.FS.booster_selector(
                X,
                y,
                candidate_features,
                categorical_features,
                params,
                select_method,
                min_candidate_features,
                blocks,
                ratio,
                seed,
                report_dir,
                distributed_and_multiprocess,
                job,
            )

        if method == "fwiz":
            selected_feature, new_X = self.FS.fwiz_selector(
                X,
                y,
                candidate_features,
            )

        if method == "iv":
            selected_feature, new_X = self.FS.iv_selector(
                X,
                y,
                candidate_features,
                categorical_features,
                numerical_features,
                part_column,
                part_values,
                handle_zero,
                bins,
                minimum,
                use_base,
                report_dir,
                distributed_and_multiprocess,
                job,
            )

        if method == "psi":
            selected_feature, new_X = self.FS.psi_selector(
                X,
                part_column,
                candidate_features,
                categorical_features,
                numerical_features,
                part_values,
                bins,
                minimal,
                priori,
                report_dir,
                distributed_and_multiprocess,
                job,
            )

        if method == "tmodel":
            selected_feature, new_X = self.FS.tmodel_selector(
                X,
                y,
                candidate_features,
                categorical_features,
                init_score,
                tmodel_method,
                params,
                importance_metric,
                folds,
                seed,
                report_dir,
                distributed_and_multiprocess,
                job,
            )

        return selected_feature, new_X
