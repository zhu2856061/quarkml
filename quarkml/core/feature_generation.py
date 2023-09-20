# -*- coding: utf-8 -*-
# @Time   : 2023/7/27 09:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function
import os
import pandas as pd
from loguru import logger
from quarkml.generator.baisc_operation import BasicGeneration
from quarkml.generator.booster import BoosterSelector
from typing import List

from quarkml.utils import tree_to_formula, transform

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


class FeatureGeneration(object):
    """ 特征衍生功能，分为两步：
           1. 基于衍生算子产生大量特征
           2. 基于提升法对特征进行筛选出有效特征
    """

    def __init__(self) -> None:
        self.bg = BasicGeneration()
        self.bs = BoosterSelector()

    def fit(
        self,
        ds: pd.DataFrame,
        label: str,
        cat_features: List = None,
        is_filter=True,
        params=None,
        select_method='predictive',
        min_candidate_features=200,
        blocks=2,
        ratio=0.5,
        distributed_and_multiprocess=-1,
        report_dir="encode",
    ):
        # step0: 划分X y
        X = ds.drop(label, axis=1)
        y = ds[[label]]

        # step1: 衍生
        candidate_features = self.basic_generation(X, cat_features, report_dir)

        if not is_filter:
            X, _ = transform(X, candidate_features)
            # 组装
            X[label] = y[label]
            return X

        # step2: 筛选
        X = self.booster_filter(
            X,
            y,
            candidate_features,
            cat_features,
            params,
            select_method,
            min_candidate_features,
            blocks,
            ratio,
            distributed_and_multiprocess,
            report_dir,
        )
        # 组装
        X[label] = y[label]

        return X

    def basic_generation(
        self,
        X: pd.DataFrame,
        cate_features: List = None,
        report_dir: str = 'encode',
    ):
        """ 基于衍生算子产生大量候选特征 """
        candidate_features = self.bg.fit(X, cate_features)

        logger.info(
            f"===== candidate_features number: {len(candidate_features)} ====="
        )
        # 存储结果
        stage1 = pd.DataFrame({'candidate_features': candidate_features})
        os.makedirs(report_dir, exist_ok=True)
        stage1.to_csv(report_dir + "/candidate_features.csv", index=False)

        return candidate_features

    def booster_filter(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        candidate_features: List[str],
        cate_features: List = None,
        params=None,
        select_method='predictive',
        min_candidate_features=200,
        blocks=5,
        ratio=0.5,
        distributed_and_multiprocess=-1,
        report_dir="encode",
    ):
        """ 基于提升法对特征进行筛选出有效特征 """
        selected_feature, new_X = self.bs.fit(
            X,
            y,
            candidate_features,
            cate_features,
            params,
            select_method,
            min_candidate_features,
            blocks,
            ratio,
            distributed_and_multiprocess,
        )

        logger.info(f"===== selected_feature: {selected_feature} =====")
        stage1 = pd.DataFrame({'selected_feature': selected_feature})
        os.makedirs(report_dir, exist_ok=True)
        stage1.to_csv(report_dir + "/selected_feature.csv", index=False)

        return new_X
