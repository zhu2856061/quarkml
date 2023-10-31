# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

from quarkml.processing.feature_processing import FeatureProcessing
from quarkml.generator.feature_generation import FeatureGeneration
from quarkml.index.index_col import FeatureIndex
from quarkml.model.model_train import TreeModel

from loguru import logger
import pandas as pd
import time
import os
import json
from typing import List
import joblib
import shap


# 模型工程
class Engine(object):

    def __init__(self) -> None:
        self.FP = FeatureProcessing()
        self.FG = FeatureGeneration()
        self.FS = FeatureIndex()
        self.TM = TreeModel()
        shap.initjs()

    def feature_processing(
        self,
        ds: pd.DataFrame,
        label: str,
        cat_feature: List = [],
        num_feature: List = [],
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

        X, cat_features, num_features = self.FP.fit(
            ds,
            label,
            cat_feature,
            num_feature,
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

    def feature_index(
        self,
        ds: pd.DataFrame,
        label: str,
        part_column: str = None,
        cate_features: List[str] = None,
        part_values: List = None,
        bins=10,
        method: str = "iv",
        distributed_and_multiprocess=-1,
        report_dir="encode",
    ):
        start = time.time()
        if isinstance(ds, str):
            ds = pd.read_csv(ds)

        if method == "iv":
            _value = self.FS.iv_index(
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
            _value = self.FS.psi_index(
                ds,
                label,
                part_column,
                cate_features,
                part_values,
                bins,
                distributed_and_multiprocess,
                report_dir,
            )

        logger.info(
            f'*************** [feature_selector] cost time: {time.time()-start} ***************'
        )
        return _value

    def model_hparams(
        self,
        ds: pd.DataFrame,
        label: str,
        valid_ds: pd.DataFrame = None,
        cat_features=None,
        params=None,
        spaces=None,
        report_dir="encode",
    ):

        start = time.time()
        if isinstance(ds, str):
            ds = pd.read_csv(ds)

        # step0: 划分X y
        X_train = ds.drop(label, axis=1)
        y_train = ds[[label]]

        if valid_ds is not None:
            if isinstance(valid_ds, str):
                valid_ds = pd.read_csv(valid_ds)

            X_test = valid_ds.drop(label, axis=1)
            y_test = valid_ds[[label]]
        else:
            X_test = None
            y_test = None

        best_params = self.TM.lgb_hparams(
            trn_x=X_train,
            trn_y=y_train,
            val_x=X_test,
            val_y=y_test,
            cat_features=cat_features,
            params=params,
            spaces=spaces,
        )

        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "hparams.json"), "w") as f:
            json.dump(best_params, f)

        logger.info(
            f'*************** [hparams] cost time: {time.time()-start} ***************'
        )

        return best_params

    def model_train(
        self,
        ds: pd.DataFrame,
        label: str,
        valid_ds: pd.DataFrame = None,
        cat_features=None,
        params=None,
        report_dir="encode",
    ):
        start = time.time()
        if isinstance(ds, str):
            ds = pd.read_csv(ds)

        # step0: 划分X y
        X_train = ds.drop(label, axis=1)
        y_train = ds[[label]]

        if valid_ds is not None:
            if isinstance(valid_ds, str):
                valid_ds = pd.read_csv(valid_ds)

            X_test = valid_ds.drop(label, axis=1)
            y_test = valid_ds[[label]]
        else:
            X_test = None
            y_test = None

        gbm, report_dict = self.TM.lgb_model(
            X_train,
            y_train,
            X_test,
            y_test,
            cat_features,
            params,
        )
        print(report_dict)
        joblib.dump(gbm, report_dir + '/loan_model.pkl')

        logger.info(
            f'*************** [hparams] cost time: {time.time()-start} ***************'
        )
        return gbm

    def interpretable(
        self,
        model,
        X: pd.DataFrame,
    ):
        
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        return shap_values

