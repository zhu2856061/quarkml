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
import joblib
from sklearn.metrics import roc_auc_score, roc_curve
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
    importance_metric: str = "importance",
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
    if _label_unique_num <= 2:
        params_set.update({'objective': 'regression'})
        params_set.update({'metric': 'auc'})
    else:
        params_set.update({'objective': 'multiclass'})
        params_set.update({'metric': 'auc_mu'})
    if cat_features is not None:
        params_set.update(
            {"categorical_feature": 'name:' + ','.join(cat_features)})
    if params is not None:
        params_set.update(params)

    logger.info(
        f"************************************ model parameters : {params_set} ************************************"
    )

    if "class" in params_set['objective'] or "binary" == params_set[
            'objective']:
        gbm = lgb.LGBMClassifier(**params_set)
    else:
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

    if importance_metric == "all" or importance_metric == "permutation":
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

    if importance_metric == "all" or importance_metric == "shap":
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

    return gbm, report_dict, params_set


def lgb_save(model, report_dir):
    joblib.dump(model, report_dir + '/loan_model.pkl')


def _auc(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    return roc_auc_score(y_true, y_scores)


def _ks(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    FPR, TPR, thresholds = roc_curve(y_true, y_scores)
    return abs(FPR - TPR).max()


def _get_categorical_numerical_features(ds: pd.DataFrame):
    # 获取类别特征，除number类型外的都是类别特征
    categorical_features = list(ds.select_dtypes(exclude=np.number))
    numerical_features = []
    for feature in ds.columns:
        if feature in categorical_features:
            continue
        else:
            numerical_features.append(feature)
    categorical_features = [str(_) for _ in categorical_features]
    numerical_features = [str(_) for _ in numerical_features]
    return categorical_features, numerical_features
