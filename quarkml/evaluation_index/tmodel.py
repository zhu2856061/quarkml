# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import os
import pickle
import numpy as np
import time
import ray
from ray.util.multiprocessing import Pool
from loguru import logger
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from quarkml.model.tree_model import lgb_train
from typing import List
from quarkml.utils import transform, error_callback
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


class TModelSelector(object):

    def __init__(self):
        pass

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        candidate_features: List = None,
        categorical_features: List = None,
        init_score: pd.DataFrame = None,
        method: str = 'mean',
        params=None,
        importance_metric: str = "importance",
        folds=5,
        seed=2023,
        report_dir: str = "encode",
        distributed_and_multiprocess=-1,
        job = -1,
    ):
        """ candidate_features : 衍生方法产生的衍生特征，将会在这里进行筛选，不为None 的话，将只对衍生特征进行筛选
            categorical_features : 类别特征list
            init_score : 对应y 大小的上一个模型预测分
            method : 对于n次交叉后，取 n次的最大，还是n次的平均，mean , max
            params : 模型参数
            importance_metric : 特征重要性判断依据 importance, permutation, shap, all
            report_dir: 将保存交叉验证后的每个特征的重要性结果
            is_distributed: 分布式采用ray进行交叉验证的每次结果，否则将进行多进程的pool池模式
        """
        start = time.time()
        if candidate_features is not None:
            raw_X = X
            X, _ = transform(X, candidate_features)

        if job < 0:
            job = os.cpu_count()

        kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

        # 【注意】这里，可以采用多进程模型 pool 池，也可以采用ray 的多进程和分布式
        if distributed_and_multiprocess == 1:
            lgb_train_remote = ray.remote(lgb_train)
        elif distributed_and_multiprocess == 2:
            pool = Pool(job)

        futures_list = []
        for i, (t_index, v_index) in enumerate(kf.split(X, y)):
            logger.info(
                f'************************************ {i + 1} ************************************'
            )

            trn_x = X.iloc[t_index]
            trn_y = y.iloc[t_index]

            val_x = X.iloc[v_index]
            val_y = y.iloc[v_index]
            if init_score is not None:
                trn_init_score = init_score.iloc[t_index]
                val_init_score = init_score.iloc[v_index]
            else:
                trn_init_score = None
                val_init_score = None

            if distributed_and_multiprocess == 1:
                futures = lgb_train_remote.remote(
                    trn_x,
                    trn_y,
                    val_x,
                    val_y,
                    categorical_features,
                    params,
                    trn_init_score,
                    val_init_score,
                    importance_metric,
                    seed,
                )
            elif distributed_and_multiprocess == 2:
                futures = pool.apply_async(lgb_train, (
                    trn_x,
                    trn_y,
                    val_x,
                    val_y,
                    categorical_features,
                    params,
                    trn_init_score,
                    val_init_score,
                    importance_metric,
                    seed,
                ), error_callback=error_callback)
            else:
                futures = lgb_train(
                    trn_x,
                    trn_y,
                    val_x,
                    val_y,
                    categorical_features,
                    params,
                    trn_init_score,
                    val_init_score,
                    importance_metric,
                    seed,
                )
            futures_list.append(futures)

        if distributed_and_multiprocess == 2:
            pool.close()
            pool.join()
        cv_auc = []
        cv_ks = []
        featrue_importance = {}
        featrue_permutation = {}
        featrue_shap = {}

        if distributed_and_multiprocess == 1:
            futures_list = [_ for _ in ray.get(futures_list)]
        elif distributed_and_multiprocess == 2:
            futures_list = [_.get() for _ in futures_list]

        for items in futures_list:
            for k, v in items[1]["featrue_importance"].items():
                try:
                    featrue_importance[k].append(v)
                except KeyError:
                    featrue_importance[k] = [v]

            for k, v in items[1]["featrue_permutation"].items():
                try:
                    featrue_permutation[k].append(v)
                except KeyError:
                    featrue_permutation[k] = [v]

            for k, v in items[1]["featrue_shap"].items():
                try:
                    featrue_shap[k].append(v)
                except KeyError:
                    featrue_shap[k] = [v]

            if items[1]["auc_score"] is not None:
                cv_auc.append(items[1]["auc_score"])

            if items[1]["ks_score"] is not None:
                cv_ks.append(items[1]["ks_score"])

        logger.info(
            f'******* end tmodel total time: {time.time()-start} *******')
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "tmodel.pkl"), "wb") as f:
            pickle.dump(
                {
                    "featrue_importance": featrue_importance,
                    "featrue_permutation": featrue_permutation,
                    "featrue_shap": featrue_shap,
                }, f)

        if method == 'max':
            idx = cv_auc.index(max(cv_auc))
            for k, v in featrue_importance.items():
                featrue_importance[k] = v[idx]

            for k, v in featrue_permutation.items():
                featrue_permutation[k] = v[idx]

            for k, v in featrue_shap.items():
                featrue_shap[k] = v[idx]

        if method == 'mean':
            for k, v in featrue_importance.items():
                featrue_importance[k] = np.mean(v)

            for k, v in featrue_permutation.items():
                featrue_permutation[k] = np.mean(v)

            for k, v in featrue_shap.items():
                featrue_shap[k] = np.mean(v)

        featrue_importance = sorted(featrue_importance.items(),
                                    key=lambda _: _[1],
                                    reverse=True)
        featrue_permutation = sorted(featrue_permutation.items(),
                                     key=lambda _: _[1],
                                     reverse=True)
        featrue_shap = sorted(featrue_shap.items(),
                              key=lambda _: _[1],
                              reverse=True)

        selected_fea = [k for k, v in featrue_importance if v > 0]

        if importance_metric == "permutation":
            selected_fea = [k for k, v in featrue_permutation if v > 0]

        if importance_metric == "shap":
            selected_fea = [k for k, v in featrue_shap if v > 0]

        if importance_metric == "all":
            selected_fea = [
                _ for _ in selected_fea
                if _ in [k for k, v in featrue_permutation if v > 0]
            ]
            selected_fea = [
                _ for _ in selected_fea
                if _ in [k for k, v in featrue_shap if v > 0]
            ]

        if candidate_features is not None:
            cand_selected_fea = []
            for k in selected_fea:
                if "booster_f_" in k:
                    indx = int(k.replace("booster_f_", ""))
                    cand_selected_fea.append(candidate_features[indx])

            new_X, _ = transform(raw_X, cand_selected_fea)
            return cand_selected_fea, new_X

        return selected_fea, X[selected_fea]
