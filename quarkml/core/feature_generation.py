# -*- coding: utf-8 -*-
# @Time   : 2023/7/27 09:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function
import pandas as pd
from loguru import logger
from quarkml.generator.baisc_operation import BasicGeneration
from quarkml.generator.booster import BoosterSelector
from typing import List

from quarkml.utils import tree_to_formula

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
        candidate_features= self.basic_generation(X, report_dir)

        # step2: 筛选
        new_X = self.booster_filter(
            X,
            y,
            candidate_features,
            params,
            select_method,
            min_candidate_features,
            blocks,
            ratio,
            distributed_and_multiprocess,
            report_dir,
        )
        # 组装
        new_X[label] = y[label]

        return new_X


    def basic_generation(
        self,
        X: pd.DataFrame,
        report_dir: str = 'encode',
    ):
        """ 基于衍生算子产生大量候选特征 """
        candidate_features = self.bg.fit(X=X, report_dir=report_dir)

        logger.info(
            f"===== candidate_features number: {len(candidate_features)} ====="
        )
        # 存储结果
        stage1 = pd.DataFrame({'candidate_features': candidate_features})
        stage1.to_csv(report_dir + "/candidate_features.csv", index=False)

        return candidate_features

    def booster_filter(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        candidate_features: List[str],
        params=None,
        select_method='predictive',
        min_candidate_features=200,
        blocks=2,
        ratio=0.5,
        distributed_and_multiprocess=-1,
        report_dir="encode",
    ):
        """ 基于提升法对特征进行筛选出有效特征 """
        selected_feature, candidate_features_scores, new_X = self.bs.fit(
            X,
            y,
            candidate_features,
            params,
            select_method,
            min_candidate_features,
            blocks,
            ratio,
            distributed_and_multiprocess,
        )

        logger.info(
            f"===== selected_feature: {selected_feature} ====="
        )
        stage1_dic = {'stage1': [],'score': []}
        for fea, sc in candidate_features_scores:
            stage1_dic['stage1'].append(tree_to_formula(fea))
            stage1_dic['score'].append(sc)
        stage1 = pd.DataFrame(stage1_dic)
        stage1.to_csv(report_dir + "/candidate_features_scores.csv", index=False)

        stage2 = pd.DataFrame({'selected_feature': selected_feature})
        stage2.to_csv(report_dir + "/selected_feature.csv", index=False)

        return new_X
