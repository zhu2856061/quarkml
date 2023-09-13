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
from quarkml.utils import get_categorical_numerical_features
import pandas as pd
import os
import pickle
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

    def model_cv(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        categorical_features=None,
        params=None,
        folds=5,
        seed=2023,
        distributed_and_multiprocess=-1,
        job = -1,
    ):

        return self.TM.lgb_model_cv(
            X_train,
            y_train,
            X_test,
            y_test,
            categorical_features,
            params,
            folds,
            seed,
            distributed_and_multiprocess,
            job,
        )

    def model(
        self,
        x_train,
        y_train,
        X_test=None,
        y_test=None,
        categorical_features=None,
        params=None,
        seed=2023,
        report_dir="encode"
    ):

        tmodel = self.TM.lgb_model(
            x_train,
            y_train,
            X_test,
            y_test,
            categorical_features,
            params,
            seed,
        )
        joblib.dump(tmodel, report_dir + '/loan_model.pkl')

    def hparams(self,
        trn_x: pd.DataFrame,
        trn_y: pd.DataFrame,
        val_x: pd.DataFrame = None,
        val_y: pd.DataFrame = None,
        params=None,
        spaces=None,
        report_dir="encode"):

        best_params = self.HM.hyperopt_fit(trn_x, trn_y, val_x, val_y, params, spaces)

        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "hparams.pkl"), "wb") as f:
            pickle.dump(
                {
                    "best_params": best_params,
                }, f)
        return best_params

    def interpretable(
        self,
        model,
        X: pd.DataFrame,
    ):
        self.FI.init_set_model_x(model, X)

    def dist_model(
        self,
        trn_ds,
        label_name,
        val_ds=None,
        categorical_features = None,
        params=None,
        seed=2023,
        num_workers=2,
        trainer_resources={"CPU": 4},
        resources_per_worker={"CPU": 2},
        use_gpu=False,
        report_dir = './encode/dist_model',
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