# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import os
from typing import List
import pandas as pd
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
        file_path = None
        if isinstance(ds, str):
            file_path = ds
            ds = pd.read_csv(ds)

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

        # 写入原始路径
        if file_path:
            new_file = file_path + "_data_processing.csv"
            X.to_csv(new_file, index=False)
            return new_file

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
        file_path = None
        if isinstance(ds, str):
            file_path = ds
            ds = pd.read_csv(ds)

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

        # 写入原始路径
        if file_path:
            new_file = file_path + "_data_processing.csv"
            X.to_csv(new_file, index=False)
            return new_file

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
        ds: pd.DataFrame,
        label: str,
        cat_features: List = None,
        is_filter=True,
        params=None,
        select_method='predictive',
        min_candidate_features=200,
        blocks=5,
        ratio=0.5,
        distributed_and_multiprocess=-1,
        report_dir="encode",
    ):
        """ 特征衍生
            入参: ds: 原始数据文件路径 或 pd.DataFrame， 文件中的label
            出参: 新的数据文件路径- 与原始数据文件一个文件夹内
        """
        start = time.time()
        file_path = None
        if isinstance(ds, str):
            file_path = ds
            ds = pd.read_csv(ds)

        # 产生新数据集
        new_ds = self.FG.fit(
            ds,
            label,
            cat_features,
            is_filter,
            params,
            select_method,
            min_candidate_features,
            blocks,
            ratio,
            distributed_and_multiprocess,
            report_dir,
        )

        logger.info(
            f'*************** [feature_generation] cost time: {time.time()-start} ***************'
        )

        # 写入原始路径
        if file_path:
            new_file = file_path + "_feature_generation.csv"
            new_ds.to_csv(new_file, index=False)
            return new_file
        else:
            return new_ds

    def feature_selector(
        self,
        ds: pd.DataFrame,
        label: str,
        part_column: str = None,
        cate_features: List[str] = None,
        part_values: List = None,
        bins=10,
        importance_metric: str = "importance",
        method: str = "fwiz",
        distributed_and_multiprocess=-1,
        report_dir="encode",
    ):
        start = time.time()
        file_path = None
        if isinstance(ds, str):
            file_path = ds
            ds = pd.read_csv(ds)

        if method == "fwiz":
           ds = self.FS.fwiz_selector(ds, label)

        if method == "iv":
            ds = self.FS.iv_selector(
                ds,
                label,
                part_column,
                cate_features,
                part_values,
                bins,
                distributed_and_multiprocess,
                report_dir,
            )

        if method == "psi":
            ds = self.FS.psi_selector(
                ds,
                label,
                part_column,
                cate_features,
                part_values,
                bins,
                distributed_and_multiprocess,
                report_dir,
            )

        if method == "tmodel":
            ds = self.FS.tmodel_selector(
                ds,
                label,
                cate_features,
                importance_metric,
                distributed_and_multiprocess,
                report_dir,
            )

        logger.info(
            f'*************** [feature_selector] cost time: {time.time()-start} ***************'
        )

        # 写入原始路径
        if file_path:
            new_file = file_path + "_feature_selector.csv"
            ds.to_csv(new_file, index=False)
            return new_file

        return ds
