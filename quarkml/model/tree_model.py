# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import numpy as np
from loguru import logger
import pandas as pd
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import time
import shap

from typing import List
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


def lgb_train(
    trn_x: pd.DataFrame,
    trn_y: pd.DataFrame,
    val_x: pd.DataFrame = None,
    val_y: pd.DataFrame = None,
    cat_features: List = None,
    params=None,
    trn_init_score: pd.DataFrame = None,
    val_init_score: pd.DataFrame = None,
    report_dir: str = 'encode',
    seed=2023,
):

    start = time.time()
    featrue_importance = {}
    featrue_permutation = {}
    featrue_shap = {}

    if cat_features is not None:
        for cate_fea in cat_features:
            try:
                trn_x[cate_fea] = trn_x[cate_fea].astype('category')
                if val_x is not None:
                    val_x[cate_fea] = val_x[cate_fea].astype('category')
            except:
                continue

    if val_x is None and trn_init_score is None:
        trn_x, val_x, trn_y, val_y = train_test_split(
            trn_x,
            trn_y,
            test_size=0.3,
            random_state=seed,
        )

    if val_x is None and trn_init_score is not None:
        trn_x, val_x, trn_y, val_y, trn_init_score, val_init_score = train_test_split(
            trn_x,
            trn_y,
            trn_init_score,
            test_size=0.3,
            random_state=seed,
        )

    # defualt config
    params_set = {
        'objective': 'regression',
        'n_estimators': 2000,
        'boosting_type': 'gbdt',
        'importance_type': 'gain',
        'metric': 'auc',
        'num_leaves': 2**3,
        'max_depth': 3,
        'min_data_in_leaf': 5,
        'min_gain_to_split': 0,
        'lambda_l1': 2,
        'lambda_l2': 2,
        'bagging_freq': 3,
        'bagging_fraction': 0.7,
        'learning_rate': 0.1,
        'seed': seed,
        'feature_pre_filter': False,
        'verbosity': -1,
        'period': 100,
        'stopping_rounds': 200,
    }
    _label_unique_num = trn_y[trn_y.columns[0]].nunique()
    if _label_unique_num != 2:
        raise ValueError(f"label value is not 0 or 1")

    if params is not None:
        params_set.update(params)

    logger.info(
        f"************************************ model parameters : {params_set} ************************************"
    )

    gbm = lgb.LGBMRegressor(**params_set)
    callbacks = [
        lgb.log_evaluation(period=params_set['period']),
        lgb.early_stopping(stopping_rounds=params_set['stopping_rounds'])
    ]

    gbm.fit(
        trn_x,
        trn_y,
        init_score=trn_init_score,
        eval_init_score=[trn_init_score, val_init_score],
        eval_set=[(trn_x, trn_y), (val_x, val_y)],
        callbacks=callbacks,
    )

    # model importances
    for k, v in zip(trn_x.columns, gbm.feature_importances_):
        featrue_importance[k] = v

    # model permutation
    r = permutation_importance(
        gbm,
        val_x,
        val_y,
        n_repeats=5,
        random_state=seed,
    )
    for k, v in zip(trn_x.columns, r.importances_mean):
        featrue_permutation[k] = v

    # model shap
    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer.shap_values(trn_x)
    shap_values = np.abs(np.sum(shap_values, axis=1))
    for k, v in zip(trn_x.columns, shap_values):
        try:
            featrue_shap[k].append(v)
        except KeyError:
            featrue_shap[k] = [v]

    report_dict = {
        'featrue_importance': featrue_importance,
        'featrue_permutation': featrue_permutation,
        'featrue_shap': featrue_shap,
    }

    logger.info(
        f'************************************ end lgb total time: {time.time()-start} ************************************'
    )

    return gbm, report_dict
