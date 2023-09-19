# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import os
from typing import List
import pandas as pd
import pickle
from loguru import logger
import time

from quarkml.core.data_processing import DataProcessing
from quarkml.core.distributed_data_processing import DistDataProcessing
from quarkml.core.feature_generation import FeatureGeneration
from quarkml.core.feature_selector import FeatureSelector


# 特征工程
class FeatureEngineering(object):

    def __init__(self) -> None:
        self.DP = DataProcessing()
        self.DDP = DistDataProcessing()
        self.FG = FeatureGeneration()
        self.FS = FeatureSelector()

    def data_processing_fit(
        self,
        ds: pd.DataFrame,
        label: str,
        cat_feature: List = [],
        num_feature: List = [],
        ordinal_number=100,
        is_fillna=False,
        drop_outliers=False,
        is_token=True,
        verbosity=False,
        compress=False,
        report_dir="./encode",
    ):
        start = time.time()
        X, cat_features, num_features = self.DP.fit(
            ds,
            label,
            cat_feature,
            num_feature,
            ordinal_number,
            is_fillna,
            drop_outliers,
            is_token,
            verbosity,
            compress,
            report_dir,
        )

        logger.info(
            f'*************** [data_processing_fit] cost time: {time.time()-start} ***************'
        )
        return X, cat_features, num_features

    def data_processing_transform(
        self,
        ds: pd.DataFrame,
        label: str,
        verbosity=False,
        compress=False,
        report_dir="./encode",
    ):
        start = time.time()
        X, cat_features, num_features = self.DP.tranform(
            ds,
            label,
            verbosity,
            compress,
            report_dir,
        )

        logger.info(
            f'*************** [data_processing_transform] cost time: {time.time()-start} ***************'
        )
        return X, cat_features, num_features

    def dist_data_processing(
        self,
        files=None,
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
        file_path: str,
        label: str,
        params=None,
        select_method='predictive',
        min_candidate_features=200,
        blocks=5,
        ratio=0.5,
        distributed_and_multiprocess=-1,
        report_dir="encode",
    ):
        """ 特征衍生
            入参: file_path: 原始数据文件路径， 文件中的label
            出参: 新的数据文件路径- 与原始数据文件一个文件夹内
        """

        ds = pd.read_csv(file_path)
        # 产生新数据集
        new_ds = self.FG.fit(
            ds,
            label,
            params,
            select_method,
            min_candidate_features,
            blocks,
            ratio,
            distributed_and_multiprocess,
            report_dir,
        )
        # 写入原始路径
        new_file = file_path + "_feature_generation.csv"
        new_ds.to_csv(new_file, index=False)
        return new_file

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
