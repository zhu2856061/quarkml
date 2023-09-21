# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import os
import pandas as pd
from loguru import logger
import numpy as np
import ray
from ray import data as rdata
from ray.data.preprocessors import Categorizer
from pyarrow import csv

import pickle
from typing import List
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


class DistDataProcessing(object):

    def __init__(self) -> None:
        pass
    def fit(
        self,
        files: List,
        label_name: str,
        delimiter: str = ',',
        cat_feature: List = [],
        num_feature: List = [],
        ordinal_number=100,
        report_dir="./encode",
    ):
        # 【留意】只针对csv 数据
        parse_options = csv.ParseOptions(delimiter=delimiter)
        ds = rdata.read_csv(files, parse_options=parse_options)

        ds, categorical_features, numerical_features = self._split_categorical_numerical_features(
            ds,
            label_name,
            cat_feature,
            num_feature,
            ordinal_number,
        )

        categorical_features.sort()
        numerical_features.sort()

        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "dist_data_processing.pkl"), "wb") as f:
            pickle.dump(
                {
                    'categorical_features': categorical_features,
                    'numerical_features': numerical_features,
                }, f)

        # if categorical_features is not None and len(categorical_features) > 0:
        #     categorizer = Categorizer(columns=categorical_features)
        #     ds = categorizer.fit_transform(ds)
        #     return ds, categorical_features, numerical_features, categorizer

        return ds, categorical_features, numerical_features

    def from_pandas(self, ds: pd.DataFrame, cat_feature: List = None):
        ds = rdata.from_pandas(ds)
        if cat_feature is not None and len(cat_feature) > 0:
            categorizer = Categorizer(columns=cat_feature)
            ds = categorizer.fit_transform(ds)
        return ds

    def _split_categorical_numerical_features(
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

        # encode_feature_uniques = {}
        # for cate_fea in categorical_features:
        #     encode_feature_uniques[cate_fea] = {
        #         v: k
        #         for k, v in enumerate(ds.unique(cate_fea))
        #     }
        #     ds = ds.map_batches(self._encode_feature_uniques,
        #                         fn_kwargs={
        #                             "name": cate_fea,
        #                             "classes": encode_feature_uniques[cate_fea],
        #                         },
        #                         compute=rdata.ActorPoolStrategy(),
        #                         batch_format="pandas")

        # for num_fea in numerical_features:
        #     ds = ds.map_batches(self._numerical_features_fillna,
        #                         fn_kwargs={
        #                             "name": num_fea,
        #                         },
        #                         compute=rdata.ActorPoolStrategy(),
        #                         batch_format="pandas")

        return ds, categorical_features, numerical_features

    def _encode_feature_uniques(self, ds, name, classes):
        ds[name] = ds[name].map(classes)
        return ds

    def _get_categorical_features(self, data):
        # 获取类别特征，除number类型外的都是类别特征
        categorical_features = []
        for i, t in enumerate(data.schema().types):
            if str(t) == 'string':
                categorical_features.append(data.schema().names[i])

        return categorical_features

    def _get_ordinal_features(self,
                              data,
                              categorical_features=[],
                              ordinal_number=100):
        # 获取有序特征为 number类型且unique数 <= ordinal_number
        ordinal_features = []
        for feature in data.schema().names:
            if feature in categorical_features:
                continue
            elif len(data.unique(feature)) <= ordinal_number:
                ordinal_features.append(feature)
        return ordinal_features

    def _get_numerical_features(self,
                                data,
                                categorical_features=[],
                                ordinal_features=[]):
        # 获取 数值特征
        numerical_features = []
        for feature in data.schema().names:
            if feature in categorical_features:
                continue
            elif feature in ordinal_features:
                continue
            else:
                numerical_features.append(feature)

        return numerical_features

    def _numerical_features_fillna(self, ds, name):
        # 按照平均数填充数值型特征
        ds[name] = ds[name].astype(float)
        return ds
