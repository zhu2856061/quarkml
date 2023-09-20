# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

from quarkml.core.data_processing import DataProcessing
from quarkml.core.model_train import TreeModel
from quarkml.core.model_hparams import HparamModel
from quarkml.core.predict_tools import Predict
from loguru import logger
from quarkml.core.model_interpretable import ModelInterpretable
import pandas as pd
import time
import os
import json
from typing import List
import joblib


# 模型工程
class ModelEngineering(object):

    def __init__(self) -> None:
        self.DP = DataProcessing()
        self.TM = TreeModel()
        self.FI = ModelInterpretable()
        self.HM = HparamModel()
        self.PRED = Predict()

    def hparams(
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

        best_params = self.HM.fit(
            trn_x=X_train,
            trn_y=y_train,
            val_x=X_test,
            val_y=y_test,
            cat_feature=cat_features,
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

    def model_cv(
        self,
        ds: pd.DataFrame,
        label: str,
        valid_ds: pd.DataFrame = None,
        cat_features=None,
        params=None,
        folds=5,
        distributed_and_multiprocess=-1,
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

        score = self.TM.lgb_model_cv(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cat_features=cat_features,
            params=params,
            folds=folds,
            distributed_and_multiprocess=distributed_and_multiprocess,
        )

        logger.info(
            f'*************** [hparams] cost time: {time.time()-start} ***************'
        )
        return score

    def model(
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

        tmodel = self.TM.lgb_model(
            X_train,
            y_train,
            X_test,
            y_test,
            cat_features,
            params,
        )
        joblib.dump(tmodel, report_dir + '/loan_model.pkl')

        logger.info(
            f'*************** [hparams] cost time: {time.time()-start} ***************'
        )

        return tmodel

    def interpretable(
        self,
        task,
        model,
        X: pd.DataFrame,
        single_index: int = -1,
        muli_num: int = -1,
        is_importance=False,
    ):
        self.FI.init_set_model_x(model, X)

        if single_index > -1:
            self.FI.single_prediction(single_index, task)

        if muli_num > -1:
            self.FI.many_prediction(muli_num, task)

        if is_importance:
            self.FI.feature_dependence(task)

    def dist_model(
        self,
        trn_ds,
        label_name,
        val_ds=None,
        categorical_features=None,
        params=None,
        seed=2023,
        num_workers=2,
        trainer_resources={"CPU": 4},
        resources_per_worker={"CPU": 2},
        use_gpu=False,
        report_dir='./encode/dist_model',
    ):
        tmodel = self.TM.lgb_distributed_train(
            trn_ds,
            label_name,
            val_ds,
            categorical_features,
            params,
            seed,
            num_workers,
            trainer_resources,
            resources_per_worker,
            use_gpu,
            report_dir,
        )

    def predict_2_so(self, model_path):
        # 目前只能是lightgbm
        self.PRED.compile_model(model_path)

    def predict_load_so(self, model_so_path):
        self.PRED.load_model_so(model_so_path)

    def predict_x(self, x):
        self.PRED.predict_x(x)