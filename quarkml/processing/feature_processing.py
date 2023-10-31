# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import os
import pandas_profiling
import pandas as pd
from loguru import logger
import numpy as np
import pickle
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


class FeatureProcessing(object):

    def __init__(self) -> None:
        pass

    def fit(
        self,
        ds: pd.DataFrame,
        label: str,
        cat_feature: List = [],
        num_feature: List = [],
        is_fillna=False,
        drop_outliers=False,
        is_token=True,
        verbosity=False,
        compress=False,
        report_dir='./encode',
    ):

        # step0
        X = ds.drop(label, axis=1)
        y = ds[[label]]

        # step1: 区分类别和数值特征
        categorical_features, numerical_features = self._get_categorical_numerical_features(
            X,
            cat_feature,
            num_feature,
        )
        categorical_features.sort()
        numerical_features.sort()

        # step2: 填充缺失值
        fillna_value = None
        if is_fillna:
            X, fillna_value = self._fillna(X, numerical_features,
                                           categorical_features)

        # step3: 去掉异常值
        if drop_outliers:
            X, fillna_value = self._del_outliers(X, numerical_features)

        # step4: 离散值token化
        tokenizer_table = None
        if is_token:
            X, tokenizer_table = self._tokenizer_categorical_features(
                X, categorical_features)

        X[label] = y[label]
        # step5: 生成报告
        if verbosity:
            self._verbosity(X, label, categorical_features, numerical_features)

        # step6: 压缩
        if compress:
            X = self._reduce_mem_usage(X)  # 压缩

        # 存储处理逻辑
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "data_processing.pkl"), "wb") as f:
            pickle.dump(
                {
                    "fillna_value": fillna_value,
                    "tokenizer_table": tokenizer_table,
                    'cat_features': categorical_features,
                    'num_features': numerical_features,
                }, f)

        
        return X, categorical_features, numerical_features

    def tranform(
        self,
        ds: pd.DataFrame,
        label_name=None,
        verbosity=False,
        compress=False,
        report_dir="./encode",
    ):

        with open(os.path.join(report_dir, "data_processing.pkl"), "rb") as f:
            dp = pickle.load(f)

        cat_features = dp['cat_features']
        num_features = dp['num_features']
        fillna_value = dp['fillna_value']
        tokenizer_table = dp['tokenizer_table']

        if fillna_value:
            for key, value in fillna_value.items():
                try:
                    ds[key] = ds[key].fillna(value)
                except:
                    continue

        if tokenizer_table:
            for key, value in tokenizer_table.items():
                ds[key] = ds[key].map(lambda _: value.get(_, None))

        if verbosity:
            self._verbosity(ds, label_name, cat_features, num_features)

        if compress:
            ds = self._reduce_mem_usage(ds)  # 压缩

        return ds, cat_features, num_features

    def _get_categorical_numerical_features(
        self,
        ds: pd.DataFrame,
        cat_feature: List = [],
        num_feature: List = [],
    ):
        """划分离散值和连续值，其中序列数值会在此阶段划分进两者
        """
        categorical_features = self._get_categorical_features(ds)
        categorical_features = list(
            set(categorical_features) | set(cat_feature))

        numerical_features = self._get_numerical_features(
            ds, categorical_features)
        numerical_features = list(set(numerical_features) | set(num_feature))

        return categorical_features, numerical_features

    def _fillna(self, ds, numerical_fea, category_fea):
        # 按照平均数填充数值型特征
        fillna_value = {}
        for num in numerical_fea:
            value = ds[num].median()
            ds[num] = ds[num].fillna(value)
            fillna_value[num] = value
        # 按照众数填充类别型特征
        for cat in category_fea:
            value = ds[cat].mode()[0]
            ds[cat] = ds[cat].fillna(value)
            fillna_value[cat] = value
        return ds, fillna_value

    def _del_outliers(self, ds, numerical_features):
        """对数据删除掉异常数据， 同时给出数据报告
        """
        for fea in numerical_features:
            ds = self._find_outliers_by_3segama(ds, fea)
            ds = self._del_outliers(ds, fea)

        return ds

    def _tokenizer_categorical_features(
        self,
        ds: pd.DataFrame,
        cat_feature: List = [],
    ):
        """划分离散值和连续值，其中序列数值会在此阶段划分进两者
        """
        tokenizer_table = {}
        for fea in cat_feature:
            ds[fea] = ds[fea].astype('category')
            # 保证NaN值在唯一性中，可设置use_na_sentinel=False
            codes, uniques = pd.factorize(ds[fea])
            uniques = {v: k for k, v in enumerate(uniques.values)}
            tokenizer_table[fea] = uniques
            ds[fea] = codes

        return ds, tokenizer_table

    def _verbosity(
        self,
        ds: pd.DataFrame,
        label_name,
        categorical_features,
        numerical_features,
    ):
        """展示数据集的相关分析指标
        """
        logger.info("============== describe ============")
        print(ds.info())
        print(ds.describe())

        logger.info("============== data category distribution ============")
        for column in categorical_features:
            self._get_cat_distribution_by_label(ds, label_name, column)

        logger.info("============== data numerical distribution ============")
        for column in numerical_features:
            self._get_num_distribution_by_label(ds, label_name, column)

        logger.info("========= data report to './report.html' ============")
        pfr = pandas_profiling.ProfileReport(ds)
        pfr.to_file(dir + "/report.html")

    def _get_categorical_features(self, data: pd.DataFrame):
        # 获取类别特征，除number类型外的都是类别特征
        return list(data.select_dtypes(exclude=np.number))

    def _get_numerical_features(self,
                                ds: pd.DataFrame,
                                categorical_features=[]):
        # 获取 数值特征
        numerical_features = []
        for feature in ds.columns:
            if feature in categorical_features:
                continue
            else:
                numerical_features.append(feature)

        return numerical_features

    def _find_outliers_by_3segama(self, ds: pd.DataFrame, fea):
        """
        # 检测异常的方法一：均方差
        # 在统计学中，如果一个数据分布近似正态，
        # 那么大约 68% 的数据值会在均值的一个标准差范围内，
        # 大约 95% 会在两个标准差范围内，
        # 大约 99.7% 会在三个标准差范围内。
        """
        data_std = np.std(ds[fea])
        data_mean = np.mean(ds[fea])
        outliers_cut_off = data_std * 3
        lower_rule = data_mean - outliers_cut_off
        upper_rule = data_mean + outliers_cut_off
        ds[fea + '_outliers'] = ds[fea].apply(lambda x: str(
            '异常值') if x > upper_rule or x < lower_rule else '正常值')

        print(ds[fea + '_outliers'].value_counts())
        print('*' * 10)
        return ds

    def _del_outliers(self, ds: pd.DataFrame, fea):
        """ 删除异常值 """
        ds = ds[ds[fea + '_outliers'] == '正常值']
        ds = ds.drop(fea + '_outliers', axis=1)
        ds = ds.reset_index(drop=True)
        return ds

    def _get_cat_distribution_by_label(self, ds: pd.DataFrame, label_name,
                                       column):
        """ 根绝y值不同可视化x离散特征的分布 """
        column_num = ds[column].nunique()
        if column_num > 100000:
            return
        label_values = ds[label_name].unique()
        fig, ax = plt.subplots(len(label_values), 1, figsize=(15, 20))
        for i, lv in enumerate(label_values):
            train_loan_fr = ds[[label_name, column]].loc[ds[label_name] == lv]

            train_loan_fr.groupby(column)[column].count().plot(
                kind='barh', ax=ax[i], title=f'Count of {column} label = {lv}')
        plt.savefig(f'encode/{column}.png')

    def _get_num_distribution_by_label(self, ds: pd.DataFrame, label_name,
                                       column):
        """ 根绝y值不同可视化x连续特征的分布 """
        label_values = ds[label_name].unique()
        fig, ax = plt.subplots(len(label_values), 1, figsize=(15, 20))
        for i, lv in enumerate(label_values):
            train_loan_fr = ds[[label_name, column]].loc[ds[label_name] == lv]
            train_loan_fr[column].plot(kind='hist',
                                       ax=ax[i],
                                       color='b',
                                       bins=100,
                                       title=f'Count of {column} label = {lv}')
        plt.savefig(f'encode/{column}.png')

    def _reduce_mem_usage(self, df, verbose=True):
        # 对类型进行压缩
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                            np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                            np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                            np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                            np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                            np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                            np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) /
                                            start_mem))

        return df
