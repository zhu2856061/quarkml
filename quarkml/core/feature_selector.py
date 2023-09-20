# -*- coding: utf-8 -*-
# @Time   : 2023/7/27 09:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import os
import pandas as pd
from loguru import logger
import time
from featurewiz import FeatureWiz
from quarkml.selector.woe_iv import WOEIV
from quarkml.selector.psi import PSI
from quarkml.selector.tmodel import TModelSelector
from quarkml.utils import transform
from typing import List, Dict
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


class FeatureSelector(object):

    def __init__(self) -> None:
        self.IVObj = WOEIV()
        self.PSIObj = PSI()
        self.TModelObj = TModelSelector()

    def fwiz_selector(
        self,
        ds: pd.DataFrame,
        label: str,
        corr_limit=0.99,
        nrows=None,
    ):
        """ 基于SULOV（搜索不相关的变量列表），SULOV 注定只能针对连续值，
        SULOV算法基于本文中解释的最小冗余最大相关性（MRMR）算法，该算法是最佳特征选择方法之一，
        详细逻辑：
        1. 查找超过相关阈值（例如corr_limit （0.99））的所有高度相关变量对
        2. 然后将他们的 MIS 分数（互助信息分数）找到目标变量。MIS 是一种非参数评分方法。所以它适用于各种变量和目标
        3. 现在取每对相关变量，然后剔除MIS分数较低的变量，剩下的是信息得分最高且彼此相关性最小的那些
        4. 递归XGBoost的工作原理如下：一旦SULOV选择了互信息得分高且相关性最小的变量，
           featurewiz使用XGBoost在SULOV之后的剩余变量中反复找到最佳特征
        """
        fwiz = FeatureWiz(
            corr_limit=corr_limit,
            nrows=nrows,
        )
        # step0: 划分X y
        X = ds.drop(label, axis=1)
        y = ds[[label]]
        fwiz_selector = fwiz.fit(X, y)

        logger.info(
            f"=== current_feature number: {len(X.columns)}, selected feature number: {len(fwiz_selector.features)} ==="
        )
        X=X[fwiz_selector.features]
        X[label] = y[label]
        return X

    def iv_selector(
        self,
        ds: pd.DataFrame,
        label: str,
        part_column: str = None,
        cat_features: List = None,
        part_values: List = None,
        bins=10,
        distributed_and_multiprocess=-1,
        report_dir='encode',
    ):
        # step0: 划分X y
        X = ds.drop(label, axis=1)
        y = ds[[label]]
        """ 基于IV值 大于 0.02 为有效特征进行选择 """
        selected_fea, X, woe, iv = self.IVObj.fit(
            X=X,
            y=y,
            part_column=part_column,
            cat_features=cat_features,
            part_values=part_values,
            bins=bins,
            distributed_and_multiprocess=distributed_and_multiprocess,
        )
        logger.info(
            f"=== current_feature number: {len(X.columns)}, selected feature number: {len(selected_fea)} ==="
        )

        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(report_dir + "/woe", exist_ok=True)
        for name, ds in woe.items():
            ds.to_csv(report_dir + "/woe/" + name + "_woe.csv")
        tmp = sorted(iv.items(), key=lambda _: _[1], reverse=True)
        iv = pd.DataFrame(tmp)
        iv.to_csv(report_dir + "/iv.csv")

        X[label] = y[label]
        return X

    def psi_selector(
        self,
        ds: pd.DataFrame,
        label: str,
        part_column: str,
        cat_features: List = None,
        part_values: List = None,
        bins: int = 10,
        distributed_and_multiprocess=-1,
        report_dir='encode',
    ):
        # step0: 划分X y
        X = ds.drop(label, axis=1)
        y = ds[[label]]

        selected_fea, X, psi_detail, psi = self.PSIObj.fit(
            X=X,
            part_column=part_column,
            cat_features=cat_features,
            part_values=part_values,
            bins=bins,
            distributed_and_multiprocess=distributed_and_multiprocess,
        )

        logger.info(
            f"=== current_feature number: {len(X.columns)}, selected feature number: {len(selected_fea)} ==="
        )
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(report_dir + "/psi", exist_ok=True)
        for name, ds in psi_detail.items():
            ds.to_csv(report_dir + "/psi/" + name + "_psi.csv")
        psi.to_csv(report_dir + "/psi.csv")

        X[label] = y[label]
        return ds

    def tmodel_selector(
        self,
        ds: pd.DataFrame,
        label: str,
        cat_features: List = None,
        importance_metric: str = "importance",
        distributed_and_multiprocess=-1,
        report_dir='encode',
    ):
        """ categorical_features : 类别特征list
            init_score : 对应y 大小的上一个模型预测分
            method : 对于n次交叉后，取 n次的最大，还是n次的平均，mean , max
            params : 模型参数
            importance_metric : 特征重要性判断依据 importance, permutation, shap, all
        """
        # step0: 划分X y
        X = ds.drop(label, axis=1)
        y = ds[[label]]

        selected_fea, X, score  = self.TModelObj.fit(
            X=X,
            y=y,
            cat_features=cat_features,
            importance_metric=importance_metric,
            distributed_and_multiprocess=distributed_and_multiprocess,
        )

        logger.info(
            f"=== current_feature number: {len(X.columns)}, selected feature number: {len(selected_fea)} ==="
        )

        iv = pd.DataFrame(score)
        iv.to_csv(report_dir + "/importance_score.csv")

        X[label] = y[label]
        return ds
