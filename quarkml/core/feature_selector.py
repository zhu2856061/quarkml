# -*- coding: utf-8 -*-
# @Time   : 2023/7/27 09:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import pandas as pd
from loguru import logger
import time
from featurewiz import FeatureWiz
from quarkml.evaluation_index.woe_iv import WOEIV
from quarkml.evaluation_index.psi import PSI
from quarkml.evaluation_index.tmodel import TModelSelector
from quarkml.evaluation_index.booster import BoosterSelector
from quarkml.utils import transform
from typing import List, Dict
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


class FeatureSelector(object):

    def __init__(self) -> None:
        self.IVObj = WOEIV()
        self.PSIObj = PSI()
        self.TModelObj = TModelSelector()
        self.BoosterObj = BoosterSelector()

    def fwiz_selector(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        candidate_features: List = None,
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
        if candidate_features is not None:
            raw_X = X
            X, _ = transform(X, candidate_features)

        fwiz_selector = fwiz.fit(X, y)
        if candidate_features is not None:
            selected_fea = []
            for k in fwiz_selector.features:
                if "booster_f_" in k:
                    indx = int(k.replace("booster_f_", ""))
                    selected_fea.append(candidate_features[indx])

            new_X, _ = transform(raw_X, selected_fea)
            logger.info(
                f"=== current_feature number: {len(candidate_features)}, selected feature number: {len(selected_fea)} ==="
            )
            return selected_fea, new_X

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
        candidate_features: List = None,
        categorical_features: List = None,
        numerical_features: List = None,
        part_column: str = None,
        part_values: List = None,
        handle_zero="merge",
        bins=10,
        minimum=0.5,
        use_base=True,
        report_dir="encode",
        distributed_and_multiprocess=-1,
        job=-1,
    ):
        """ 基于IV值 大于 0.02 为有效特征进行选择 """
        start = time.time()
        selected_fea, ds = self.IVObj.fit(
            X,
            y,
            candidate_features,
            categorical_features,
            numerical_features,
            part_column,
            part_values,
            handle_zero,
            bins,
            minimum,
            use_base,
            report_dir,
            distributed_and_multiprocess,
            job,
        )

        if candidate_features is not None:
            logger.info(
                f"=== current_feature number: {len(candidate_features)}, selected feature number: {len(selected_fea)} ==="
            )
        else:
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
        candidate_features: List = None,
        categorical_features: List = None,
        numerical_features: List = None,
        part_values: List = None,
        bins: int = 10,
        minimal: int = 1,
        priori: Dict = None,
        report_dir="encode",
        distributed_and_multiprocess=-1,
        job=-1,
    ):
        start = time.time()
        selected_fea, ds = self.PSIObj.fit(
            X,
            part_column,
            candidate_features,
            categorical_features,
            numerical_features,
            part_values,
            bins,
            minimal,
            priori,
            report_dir,
            distributed_and_multiprocess,
            job,
        )

        if candidate_features is not None:
            logger.info(
                f"=== current_feature number: {len(candidate_features)}, selected feature number: {len(selected_fea)} ==="
            )
        else:
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
        candidate_features: List = None,
        categorical_features: List = None,
        init_score: pd.DataFrame = None,
        method: str = 'mean',
        params: Dict = None,
        importance_metric: str = "importance",
        folds=5,
        seed=2023,
        report_dir: str = "encode",
        distributed_and_multiprocess=-1,
        job=-1,
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
            candidate_features,
            categorical_features,
            init_score,
            method,
            params,
            importance_metric,
            folds,
            seed,
            report_dir,
            distributed_and_multiprocess,
            job,
        )

        if candidate_features is not None:
            logger.info(
                f"=== current_feature number: {len(candidate_features)}, selected feature number: {len(selected_fea)} ==="
            )
        else:
            logger.info(
                f"=== current_feature number: {len(X.columns)}, selected feature number: {len(selected_fea)} ==="
            )

        logger.info(
            f'************************************ [tmodel_selector] cost time: {time.time()-start} ************************************'
        )
        return selected_fea, ds

    def booster_selector(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        candidate_features: List,
        categorical_features: List = None,
        params: Dict = None,
        select_method='predictive',
        min_candidate_features=200,
        blocks=2,
        ratio=0.5,
        seed=2023,
        report_dir="encode",
        distributed_and_multiprocess=-1,
        job=-1,
    ):
        start = time.time()
        selected_fea, ds = self.BoosterObj.fit(
            X,
            y,
            candidate_features,
            categorical_features,
            params,
            select_method,
            min_candidate_features,
            blocks,
            ratio,
            seed,
            report_dir,
            distributed_and_multiprocess,
            job,
        )
        logger.info(
            f"=== candidate_feature number: {len(candidate_features)}, selected feature number: {len(selected_fea)} ==="
        )
        logger.info(
            f'************************************ [booster_selector] cost time: {time.time()-start} ************************************'
        )
        return selected_fea, ds
