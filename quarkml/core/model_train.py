# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import os
import numpy as np
import time
import ray
from ray.util.multiprocessing import Pool
from loguru import logger
from quarkml.model.tree_model import lgb_train, _auc, _ks
from quarkml.model.distributed_tree_model import lgb_distributed_train
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings('ignore')


class TreeModel(object):

    def __init__(self):
        pass

    def lgb_model(
        self,
        x_train,
        y_train,
        X_test=None,
        y_test=None,
        cat_features=None,
        params=None,
    ):

        return lgb_train(x_train,
                         y_train,
                         X_test,
                         y_test,
                         cat_features,
                         params)

    def lgb_model_cv(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        cat_features=None,
        params=None,
        folds=5,
        distributed_and_multiprocess=-1,
    ):
        kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2023)

        job = os.cpu_count() - 2

        if distributed_and_multiprocess == 1:
            lgb_train_remote = ray.remote(lgb_train)
        elif distributed_and_multiprocess == 2:
            pool = Pool(job)

        futures_list = []
        for i, (train_index,
                valid_index) in enumerate(kf.split(X_train, y_train)):
            logger.info(
                f'************************************ {i + 1} ************************************'
            )

            trn_x = X_train.iloc[train_index]
            trn_y = y_train.iloc[train_index]

            val_x = X_train.iloc[valid_index]
            val_y = y_train.iloc[valid_index]

            if distributed_and_multiprocess == 1:
                futures = lgb_train_remote.remote(
                    trn_x,
                    trn_y,
                    val_x,
                    val_y,
                    cat_features,
                    params,
                )
            elif distributed_and_multiprocess == 2:
                futures = pool.apply_async(lgb_train, (
                    trn_x,
                    trn_y,
                    val_x,
                    val_y,
                    cat_features,
                    params,
                ))
            else:
                futures = lgb_train(
                    trn_x,
                    trn_y,
                    val_x,
                    val_y,
                    cat_features,
                    params,
                )
            futures_list.append(futures)

        if distributed_and_multiprocess == 2:
            pool.close()
            pool.join()

        if distributed_and_multiprocess == 1:
            futures_list = [_ for _ in ray.get(futures_list)]
        elif distributed_and_multiprocess == 2:
            futures_list = [_.get() for _ in futures_list]

        auc_ks_result = []
        for i, items in enumerate(futures_list):
            try:
                if X_test is not None:
                    test_pred = items[0].predict(
                        X_test,
                        num_iteration=items[0]._best_iteration,
                    )
                    auc_score = _auc(y_test, test_pred)
                    ks_score = _ks(y_test, test_pred)
                else:
                    val_pred = items[0].predict(
                        val_x,
                        num_iteration=items[0]._best_iteration,
                    )
                    auc_score = _auc(val_y, val_pred)
                    ks_score = _ks(val_y, val_pred)

                logger.info(
                    f"StratifiedKFold: {i}, auc:{auc_score}, ks:{ks_score}")
                auc_ks_result.append([auc_score, ks_score])
            except:
                logger.warning("eval error")

        if len(auc_ks_result) > 0:
            cv_auc = []
            cv_ks = []
            for k, v in auc_ks_result:
                cv_auc.append(k)
                cv_ks.append(v)

            logger.info(f"scotrainre_list, auc_list:{cv_auc}, ks_list:{cv_ks}")
            logger.info(f"auc_mean:{np.mean(cv_auc)}")
            logger.info(f"auc_std:{np.std(cv_auc)}")
            logger.info(f"ks_mean:{np.mean(cv_ks)}")
            logger.info(f"ks_std:{np.std(cv_ks)}")

            return np.mean(cv_auc), np.mean(cv_ks)

        return 0, 0

    def lgb_distributed_model(
        self,
        trn_ds,
        label_name,
        val_ds=None,
        categorical_features=None,
        params=None,
        seed=2023,
        num_workers=2,
        trainer_resources=None,
        resources_per_worker=None,
        use_gpu=False,
        report_dir='./encode/dist_model',
    ):

        return lgb_distributed_train(
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
