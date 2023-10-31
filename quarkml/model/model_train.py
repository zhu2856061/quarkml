# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

from loguru import logger
from quarkml.model.tree_model import lgb_train
from quarkml.model.distributed_tree_model import lgb_distributed_train
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

    def lgb_model(
        self,
        x_train,
        y_train,
        X_test=None,
        y_test=None,
        cat_features=None,
        params=None,
    ):

        return lgb_train(
            trn_x=x_train,
            trn_y=y_train,
            val_x=X_test,
            val_y=y_test,
            cat_features=cat_features,
            params=params,
        )

    def lgb_hparams(self,
        trn_x,
        trn_y,
        val_x=None,
        val_y=None,
        cat_features=None,
        params=None,
        spaces=None,
    ):
        if val_x is None:
            trn_x, val_x, trn_y, val_y = train_test_split(
                trn_x,
                trn_y,
                test_size=0.3,
                random_state=2023,
            )

        self.params = params
        self.trn_x = trn_x
        self.val_x = val_x
        self.trn_y = trn_y
        self.val_y = val_y
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
            trn_x=self.trn_x,
            trn_y=self.trn_y,
            val_x=self.val_x,
            val_y=self.val_y,
            cat_features=self.cat_features,
            params=config,
        )

        val_pred = futures[0].predict(
            self.val_x,
            num_iteration=futures[0]._best_iteration,
        )

        val_pred = futures[0].predict(
            self.val_x,
            num_iteration=futures[0]._best_iteration,
        )
        val_score = self.auc(self.val_y, val_pred)

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