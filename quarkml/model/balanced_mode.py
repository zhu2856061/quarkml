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

from quarkml.index.woe_iv import WOEIV

from typing import List
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


def balance_mode(ds: pd.DataFrame, label, cat_features=None):
    """ ds : 数据集，其中必须有一个字段sample_type
        sample_type 只具有2个值 {0, 1} 
        0 : 开发样本
        1 :  时间外样本
    """

    # 划分 开发样本和时间外样本
    tr_ds = ds[ds['sample_type'] == 0]
    vl_ds = ds[ds['sample_type'] == 1]
    tr_ds = tr_ds.drop('sample_type', axis=1)
    vl_ds = vl_ds.drop('sample_type', axis=1)
    woe_iv = WOEIV()

    # 开发样本
    tr_X = tr_ds.drop(label, axis=1)
    tr_y = tr_ds[[label]]
    tr_value = woe_iv.fit(tr_X, tr_y, cat_features=cat_features)

    # 时间外样本
    vl_X = vl_ds.drop(label, axis=1)
    vl_y = vl_ds[[label]]
    vl_value = woe_iv.fit(vl_X, vl_y, cat_features=cat_features)
    
    # 两份IV的比较
    tr_iv = tr_value[1]
    vl_iv = vl_value[1]
    print(tr_iv)
    need_name = []
    for name_i, value_i in tr_iv.items():
        for name_j, value_j in vl_iv.items():
            if name_i == name_j and np.abs(value_i - value_j) < 0.15:
                need_name.append(name_i)

    ds = ds.drop('sample_type', axis=1)

    # 新数据
    raw_columns = list(ds.columns)
    ds = ds[need_name +[label]]
    logger.info(f"==raw columns name===>{raw_columns}<=====")
    logger.info(f"==del columns name===>{set(raw_columns)-set(need_name) - set([label])}<=====")
    return ds, need_name
