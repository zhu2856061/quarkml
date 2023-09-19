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
        X: pd.DataFrame,
        y: pd.DataFrame,
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
        start = time.time()
        fwiz = FeatureWiz(
            corr_limit=corr_limit,
            nrows=nrows,
        )

        fwiz_selector = fwiz.fit(X, y)

        logger.info(
            f"=== current_feature number: {len(X.columns)}, selected feature number: {len(fwiz_selector.features)} ==="
        )
        logger.info(
            f'************************************ [fwiz_selector] cost time: {time.time()-start} ************************************'
        )
        return fwiz_selector.features, X[fwiz_selector.features]

    def iv_selector(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        part_column: str = None,
        part_values: List = None,
        handle_zero="merge",
        bins=10,
        minimum=0.5,
        use_base=True,
        report_dir="encode",
        distributed_and_multiprocess=-1,
    ):
        """ 基于IV值 大于 0.02 为有效特征进行选择 """
        start = time.time()
        selected_fea, ds = self.IVObj.fit(
            X,
            y,
            part_column,
            part_values,
            handle_zero,
            bins,
            minimum,
            use_base,
            distributed_and_multiprocess,
        )
        os.makedirs(report_dir, exist_ok=True)
        logger.info(
            f"=== current_feature number: {len(X.columns)}, selected feature number: {len(selected_fea)} ==="
        )
        logger.info(
            f'************************************ [iv_selector] cost time: {time.time()-start} ************************************'
        )
        return selected_fea, ds

    def psi_selector(
        self,
        X: pd.DataFrame,
        part_column: str,
        part_values: List = None,
        bins: int = 10,
        minimal: int = 1,
        priori: Dict = None,
        report_dir="encode",
        distributed_and_multiprocess=-1,
    ):
        start = time.time()
        selected_fea, ds = self.PSIObj.fit(
            X,
            part_column,
            part_values,
            bins,
            minimal,
            priori,
            distributed_and_multiprocess,
        )

        logger.info(
            f"=== current_feature number: {len(X.columns)}, selected feature number: {len(selected_fea)} ==="
        )
        
        logger.info(
            f'************************************ [psi_selector] cost time: {time.time()-start} ************************************'
        )
        return selected_fea, ds

    def tmodel_selector(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        method: str = 'mean',
        params: Dict = None,
        importance_metric: str = "importance",
        folds=5,
        seed=2023,
        report_dir: str = "encode",
        distributed_and_multiprocess=-1,
    ):
        """ categorical_features : 类别特征list
            init_score : 对应y 大小的上一个模型预测分
            method : 对于n次交叉后，取 n次的最大，还是n次的平均，mean , max
            params : 模型参数
            importance_metric : 特征重要性判断依据 importance, permutation, shap, all
        """
        start = time.time()
        selected_fea, ds = self.TModelObj.fit(
            X,
            y,

            method,
            params,
            importance_metric,
            folds,
            seed,
            distributed_and_multiprocess,
        )

        logger.info(
            f"=== current_feature number: {len(X.columns)}, selected feature number: {len(selected_fea)} ==="
        )

        logger.info(
            f'************************************ [tmodel_selector] cost time: {time.time()-start} ************************************'
        )
        return selected_fea, ds
