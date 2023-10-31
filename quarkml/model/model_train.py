# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

from loguru import logger
from quarkml.model.tree_model import lgb_train
from quarkml.model.distributed_tree_model import lgb_distributed_train
from quarkml.model.balanced_mode import balance_mode
from hyperopt import fmin, hp, Trials, space_eval, tpe
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')


class TreeModel(object):

    def __init__(self):
        pass

    def auc(self, y_true, y_scores):
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        return roc_auc_score(y_true, y_scores)

    def lgb_train(
        self,
        tr_ds,
        label,
        vl_ds=None,
        cat_features=None,
        params=None,
    ):
        if vl_ds is None:
            tr_ds, vl_ds = train_test_split(
                tr_ds,
                test_size=0.3,
                random_state=2023,
            )

        tr_X = tr_ds.drop(label, axis=1)
        tr_y = tr_ds[[label]]

        vl_X = vl_ds.drop(label, axis=1)
        vl_y = vl_ds[[label]]

        return lgb_train(
            trn_x=tr_X,
            trn_y=tr_y,
            val_x=vl_X,
            val_y=vl_y,
            cat_features=cat_features,
            params=params,
        )

    def lgb_hparams(
        self,
        tr_ds,
        label,
        vl_ds=None,
        cat_features=None,
        params=None,
        spaces=None,
    ):
        if vl_ds is None:
            tr_ds, vl_ds = train_test_split(
                tr_ds,
                test_size=0.3,
                random_state=2023,
            )

        self.tr_X = tr_ds.drop(label, axis=1)
        self.tr_y = tr_ds[[label]]

        self.vl_X = vl_ds.drop(label, axis=1)
        self.vl_y = vl_ds[[label]]

        self.cat_features = cat_features

        if spaces is None:
            spaces = {
                "n_estimators": hp.choice("n_estimators", range(200, 2000)),
                "max_bin": hp.choice("max_bin", range(50, 500)),
                "learning_rate": hp.uniform("learning_rate", 0.001, 0.2),
                "boosting_type": hp.choice("boosting_type",
                                           ['gbdt', 'dart', 'rf']),
                "num_leaves": hp.choice("num_leaves", range(8, 64)),
                "max_depth": hp.choice("max_depth", range(3, 8)),
                "min_data_in_leaf": hp.choice("min_data_in_leaf",
                                              range(5, 100)),
                "lambda_l1": hp.uniform("lambda_l1", 0, 100),
                "lambda_l2": hp.uniform("lambda_l2", 0, 100),
                "bagging_fraction": hp.uniform("bagging_fraction", 0.5, 1.0),
                "bagging_freq": hp.choice("bagging_freq", range(1, 20)),
            }

        trials = Trials()
        best = fmin(fn=self._hyperopt_target_fn,
                    space=spaces,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trials)

        # 4，获取最优参数
        best_params = space_eval(spaces, best)
        logger.info(f"best_params={best_params}")
        return best_params

    def _hyperopt_target_fn(self, config):

        futures = lgb_train(
            trn_x=self.tr_X,
            trn_y=self.tr_y,
            val_x=self.vl_X,
            val_y=self.vl_y,
            cat_features=self.cat_features,
            params=config,
        )

        val_pred = futures[0].predict(
            self.vl_X,
            num_iteration=futures[0]._best_iteration,
        )

        val_pred = futures[0].predict(
            self.vl_X,
            num_iteration=futures[0]._best_iteration,
        )
        val_score = self.auc(self.vl_y, val_pred)

        return -val_score

    def lgb_distributed_model(
        self,
        trn_ds,
        label,
        valid_ds=None,
        cat_features=None,
        params=None,
        num_workers=2,
        trainer_resources=None,
        resources_per_worker=None,
        use_gpu=False,
        report_dir='./encode/dist_model',
    ):

        return lgb_distributed_train(
            trn_ds,
            label,
            valid_ds,
            cat_features,
            params,
            num_workers,
            trainer_resources,
            resources_per_worker,
            use_gpu,
            report_dir,
        )

    def lgb_balanced_train(
        x_train,
        tr_ds,
        label,
        vl_ds=None,
        cat_features=None,
        params=None,
    ):

        tr_ds, need_name = balance_mode(tr_ds, label, cat_features=None)

        if vl_ds is None:
            tr_ds, vl_ds = train_test_split(
                tr_ds,
                test_size=0.3,
                random_state=2023,
            )
        else:
            vl_ds = vl_ds.drop('sample_type', axis=1)
            # 新数据
            vl_ds = vl_ds[need_name]

        tr_X = tr_ds.drop(label, axis=1)
        tr_y = tr_ds[[label]]

        vl_X = vl_ds.drop(label, axis=1)
        vl_y = vl_ds[[label]]

        return lgb_train(
            trn_x=tr_X,
            trn_y=tr_y,
            val_x=vl_X,
            val_y=vl_y,
            cat_features=cat_features,
            params=params,
        )