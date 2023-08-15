# -*- coding: utf-8 -*-
# @Time   : 2023/8/8 15:26
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import numpy as np
from loguru import logger
from quarkml.model.tree_model import lgb_train
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from hyperopt import fmin, hp, Trials, space_eval, tpe
import optuna
from sklearn.metrics import roc_auc_score, f1_score
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


class HparamModel(object):

    def __init__(self) -> None:
        pass

    def auc(self, y_true, y_scores):
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        return roc_auc_score(y_true, y_scores)

    def f1(self, y_true, y_scores):
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        return f1_score(y_true, y_scores)

    def hyperopt_fit(
        self,
        trn_x,
        trn_y,
        val_x=None,
        val_y=None,
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

    def optuna_fit(self, trn_x, trn_y, val_x=None, val_y=None, params=None):
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

        study = optuna.create_study(direction="minimize", study_name="LGBM")
        study.optimize(self._optuna_target_fn, n_trials=100)
        # 4，获取最优参数
        logger.info(f"best_params={study.best_params}")
        return study.best_params

    def _hyperopt_target_fn(self, config):
        #
        futures = lgb_train(
            self.trn_x,
            self.trn_y,
            self.val_x,
            self.val_y,
            params=config,
        )
        val_pred = futures[0].predict(
            self.val_x,
            num_iteration=futures[0]._best_iteration,
        )

        if "class" in futures[2]['objective'] or "binary" == futures[2][
                'objective']:
            val_pred = futures[0].predict_proba(
                self.val_x,
                num_iteration=futures[0]._best_iteration,
            )
            val_pred = [np.argmax(_) for _ in val_pred]
            val_score = self.f1(self.val_y, val_pred)
        else:
            val_pred = futures[0].predict(
                self.val_x,
                num_iteration=futures[0]._best_iteration,
            )
            val_score = self.auc(self.val_y, val_pred)

        return -val_score

    def _optuna_target_fn(self, trials):

        params = {
            'n_estimators':
            trials.suggest_categorical("n_estimators", range(200, 2000)),
            'max_bin':
            trials.suggest_categorical("max_bin", range(50, 500)),
            'boosting_type':
            trials.suggest_categorical("boosting_type",
                                       ['gbdt', 'dart', 'rf']),
            'num_leaves':
            trials.suggest_categorical("num_leaves", range(8, 64)),
            'max_depth':
            trials.suggest_categorical("max_depth", range(3, 8)),
            'min_data_in_leaf':
            trials.suggest_categorical("min_data_in_leaf", range(5, 100)),
            'min_gain_to_split':
            trials.suggest_uniform("min_gain_to_split", 0.0, 1.0),
            'lambda_l1':
            trials.suggest_uniform("lambda_l1", 0, 100),
            'lambda_l2':
            trials.suggest_uniform("lambda_l2", 0, 100),
            'bagging_freq':
            trials.suggest_categorical("bagging_freq", range(1, 20)),
            'bagging_fraction':
            trials.suggest_uniform("bagging_fraction", 0.5, 1.0),
            'learning_rate':
            trials.suggest_uniform("learning_rate", 0.001, 0.2),
        }
        if self.params is not None:
            params.update(self.params)

        futures = lgb_train(
            self.trn_x,
            self.trn_y,
            self.val_x,
            self.val_y,
            params=params,
        )
        val_pred = futures[0].predict(
            self.val_x,
            num_iteration=futures[0]._best_iteration,
        )

        if "class" in futures[2]['objective'] or "binary" == futures[2][
                'objective']:
            val_pred = futures[0].predict_proba(
                self.val_x,
                num_iteration=futures[0]._best_iteration,
            )
            val_pred = [np.argmax(_) for _ in val_pred]
            val_score = self.f1(self.val_y, val_pred)
        else:
            val_pred = futures[0].predict(
                self.val_x,
                num_iteration=futures[0]._best_iteration,
            )
            val_score = self.auc(self.val_y, val_pred)

        return -val_score

        # # 5折交叉验证
        # cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2023)
        # cv_scores = np.empty(3)

        # for idx, (train_idx,
        #           test_idx) in enumerate(cv.split(self.trn_x, self.trn_y)):
        #     X_train, X_test = self.trn_x.iloc[train_idx], self.trn_x.iloc[
        #         test_idx]
        #     y_train, y_test = self.trn_y.iloc[train_idx], self.trn_y.iloc[
        #         test_idx]

        #     futures = lgb_train(
        #         X_train,
        #         y_train,
        #         X_test,
        #         y_test,
        #         None,
        #         params=params,
        #     )

        # if "class" in futures[2]['objective'] or "binary" == futures[2][
        #         'objective']:
        #     val_pred = futures[0].predict_proba(
        #         X_test,
        #         num_iteration=futures[0]._best_iteration,
        #     )
        #     val_pred = [np.argmax(_) for _ in val_pred]
        #     val_score = self.f1(y_test, val_pred)
        # else:
        #     val_pred = futures[0].predict(
        #         X_test,
        #         num_iteration=futures[0]._best_iteration,
        #     )
        #     val_score = self.auc(y_test, val_pred)

        # cv_scores[idx] = val_score

        # return -np.mean(cv_scores)
