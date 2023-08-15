# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import os
import pandas as pd
from loguru import logger
import numpy as np
import pickle
from typing import List
from quarkml.core.exploratory_tools import *
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


class DataProcessing(object):

    def __init__(self) -> None:
        self.dp = None

    def fit(
        self,
        ds: pd.DataFrame,
        label_name: str,
        cat_feature: List = [],
        num_feature: List = [],
        ordinal_number=100,
        report_dir="./encode",
        is_fillna=False,
        drop_outliers=False,
        verbosity=False,
    ):

        ds, encode_feature_uniques, categorical_features, numerical_features = self.split_categorical_numerical_features(
            ds,
            label_name,
            cat_feature,
            num_feature,
            ordinal_number,
        )

        categorical_features.sort()
        numerical_features.sort()
        ds, fillna_value = self.fillna_del_outliers_report(
            ds,
            label_name,
            categorical_features,
            numerical_features,
            is_fillna,
            drop_outliers,
            verbosity,
            report_dir,
        )

        # for cat in categorical_features:
        #     ds[cat] = ds[cat].astype('category')

        ds = reduce_mem_usage(ds)  # 压缩

        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "data_processing.pkl"), "wb") as f:
            pickle.dump(
                {
                    "fillna_value": fillna_value,
                    "encode_feature_uniques": encode_feature_uniques,
                    'categorical_features': categorical_features,
                    'numerical_features': numerical_features,
                }, f)

        return ds, categorical_features, numerical_features

    def tranform(
        self,
        ds: pd.DataFrame,
        label_name=None,
        report_dir="./encode",
        verbosity=False,
    ):
        if self.dp is None:
            with open(os.path.join(report_dir, "data_processing.pkl"), "rb") as f:
                self.dp = pickle.load(f)
        categorical_features = []
        numerical_features = []
        
        for cat in self.dp['categorical_features']:
            try:
                ds[cat] = ds[cat].astype('category')
                ds[cat] = ds[cat].map(lambda _: self.dp['encode_feature_uniques'][cat].get(_, None))
                ds[cat] = ds[cat].fillna(self.dp['fillna_value'][cat])
                categorical_features.append(cat)
            except:
                logger.info(f"=== del categorical_features: {cat} ===")
                continue

        for num in self.dp['numerical_features']:
            try:
                ds[num] = ds[num].fillna(self.dp['fillna_value'][num])
                ds[num] = ds[num].astype(float)
                numerical_features.append(num)
            except:
                logger.info(f"=== del numerical_features: {cat} ===")
                continue

        ds = reduce_mem_usage(ds)  # 压缩

        if verbosity:
            self.verbosity(self, ds, label_name, categorical_features,
                    numerical_features, report_dir)

        return ds

    def split_categorical_numerical_features(
        self,
        ds: pd.DataFrame,
        label_name: str,
        cat_feature: List = [],
        num_feature: List = [],
        ordinal_number=100,
    ):
        """划分离散值和连续值，其中序列数值会在此阶段划分进两者
        """
        categorical_features = self._get_categorical_features(ds)
        categorical_features = list(
            set(categorical_features) | set(cat_feature))
        ordinal_features = self._get_ordinal_features(ds, categorical_features,
                                                      ordinal_number)
        ordinal_features = list(set(ordinal_features) - set(num_feature))

        categorical_features = categorical_features + ordinal_features
        numerical_features = self._get_numerical_features(
            ds, categorical_features, ordinal_features)
        numerical_features = list(set(numerical_features) | set(num_feature))

        # 去label
        categorical_features = [
            _ for _ in categorical_features if _ != label_name
        ]
        numerical_features = [_ for _ in numerical_features if _ != label_name]
        # 对数据进行类型转换并数值化， 其中的 ordinal_features 进行指定为categorical_features
        label = ds[label_name]
        ds = ds[categorical_features + numerical_features]
        encode_feature_uniques = {}
        for cate_fea in categorical_features:
            ds[cate_fea] = ds[cate_fea].astype('category')
            # 保证NaN值在唯一性中，可设置use_na_sentinel=False
            codes, uniques = pd.factorize(ds[cate_fea])
            uniques = {v: k for k, v in enumerate(uniques.values)}
            encode_feature_uniques[cate_fea] = uniques
            ds[cate_fea] = codes

        for num_fea in numerical_features:
            ds[num_fea] = ds[num_fea].astype(float)

        ds[label_name] = label
        return ds, encode_feature_uniques, categorical_features, numerical_features

    def fillna_del_outliers_report(
        self,
        ds,
        label_name,
        categorical_features,
        numerical_features,
        is_fillna=True,
        drop_outliers=False,
        verbosity=False,
        report_dir = 'encode',
    ):
        """对数据中的None 进行填充，并且删除掉异常数据， 同时给出数据报告
        """
        fillna_value = None
        if is_fillna:
            ds, fillna_value = fillna(ds, numerical_features, categorical_features)
        if drop_outliers:
            for fea in numerical_features:
                ds = find_outliers_by_3segama(ds, fea)
                ds = del_outliers(ds, fea)

        if verbosity:
            self.verbosity(ds, label_name, categorical_features,
                           numerical_features, report_dir)
        return ds, fillna_value

    def verbosity(self, ds, label_name, categorical_features,
                  numerical_features, report_dir):
        """展示数据集的相关分析指标
        """
        logger.info("============== describe ============")
        print(get_info(ds))
        print(get_describe(ds))
        logger.info("============== data category distribution ============")
        for column in categorical_features:
            get_category_distribution_by_label(ds, label_name, column)

        logger.info("============== data numerical distribution ============")
        for column in numerical_features:
            get_numerical_serial_distribution_by_label(ds, label_name, column)
        logger.info(
            "============== data report to './report.html' ============")
        get_report(ds, report_dir)

    def _get_categorical_features(self, data: pd.DataFrame):
        # 获取类别特征，除number类型外的都是类别特征
        return list(data.select_dtypes(exclude=np.number))

    def _get_ordinal_features(self,
                              ds: pd.DataFrame,
                              categorical_features=[],
                              ordinal_number=100):
        # 获取有序特征为 number类型且unique数 <= ordinal_number
        ordinal_features = []
        for feature in ds.columns:
            if feature in categorical_features:
                continue
            elif ds[feature].nunique() <= ordinal_number:
                ordinal_features.append(feature)
        return ordinal_features

    def _get_numerical_features(self,
                                ds: pd.DataFrame,
                                categorical_features=[],
                                ordinal_features=[]):
        # 获取 数值特征
        numerical_features = []
        for feature in ds.columns:
            if feature in categorical_features:
                continue
            elif feature in ordinal_features:
                continue
            else:
                numerical_features.append(feature)

        return numerical_features
