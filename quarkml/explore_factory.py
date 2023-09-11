# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

from sklearn.preprocessing import LabelEncoder
import matplotlib
import warnings

# 让后台不显示警告
warnings.filterwarnings(action='ignore', category=UserWarning)
# 让后台不显示图 ， 想显示图 matplotlib.use('TKAgg') # jupyter 中 %matplotlib inline
matplotlib.use('Agg')


########################################################################
# 数据分析 - 第一步
# *了解数据的基础情况
########################################################################
# 数据报告
def get_report(ds: pd.DataFrame, dir: str):
    """会以html的形式展示数据基础信息"""
    import pandas_profiling
    pfr = pandas_profiling.ProfileReport(ds)
    pfr.to_file(dir + "/report.html")


# 透视图
def pivot_table(ds: pd.DataFrame, index, values, aggfunc):
    # index 是groupby 的列，可以多列['A', 'B']
    # values 是 展示的列，可以多列['C', 'D']
    pivot = pd.pivot_table(ds, index=index, values=values, aggfunc=aggfunc)
    return pivot


# 查看数据集的样本个数和原始特征维度
def get_shape(ds: pd.DataFrame):
    return ds.shape


# 获取列名
def get_columns(ds: pd.DataFrame):
    return ds.columns


# 数据类型信息
def get_info(ds: pd.DataFrame):
    return ds.info()


# 总体粗略的查看数据集各个特征的一些基本统计量
def get_describe(ds: pd.DataFrame):
    return ds.describe()


########################################################################
# 数据分析 - 第二步
# *缺失值
########################################################################
# 查看数据集中有特征缺失值的列数
def get_isnull_column_num(ds: pd.DataFrame):
    return ds.isnull().any().sum()


# 查看数据集中哪些列特征缺失值的数量
def display_missing(ds: pd.DataFrame):
    for col in ds.columns.tolist():
        print(f'{col} column missing values: {ds[col].isnull().sum()}')


# 进一步查看特征中缺失率
def get_column_missing_rate(ds: pd.DataFrame):
    for col in ds.columns.tolist():
        rate = ds[col].isnull().sum() / ds.shape[0]
        print(f'{col} column missing rate: {rate}')


# 数值型特征 - 按照平均数填充
def num_fillna(ds: pd.DataFrame, col_name):
    value = ds[col_name].median()
    ds[col_name] = ds[col_name].fillna(value)
    return ds


# 类别型特征 - 按照众数填充
def cat_fillna(ds: pd.DataFrame, col_name):
    value = ds[col_name].mode()[0]
    ds[col_name] = ds[col_name].fillna(value)
    return ds

# 分析某特征与某特征之间的占比情况，才能更好的进行填充
# 数值型特征 - 按照某特征分组，组内均值填充
def num_groupby_fillna(ds: pd.DataFrame, groupby, col_name):
    # groupby_value = ds.groupby(['Sex', 'Pclass']).median()['Age']
    groupby_value = ds.groupby(groupby).median()[col_name]
    print(groupby_value)
    ds[col_name] = ds.groupby(groupby)[col_name].apply(lambda _: _.fillna(_.median()))

# 类别型特征 - 按照某特征分组，组内众数填充
def num_groupby_fillna(ds: pd.DataFrame, groupby, col_name):
    # groupby_value = ds.groupby(['Sex', 'Pclass']).median()['Age']
    groupby_value = ds.groupby(groupby).mode()[col_name]
    print(groupby_value)
    ds[col_name] = ds.groupby(groupby)[col_name].apply(lambda _: _.fillna(_.mode()[0]))


# 查看哪些特征只有一个值的
def get_one_value_feature(ds):
    one_value_fea = [col for col in ds.columns if ds[col].nunique() <= 1]
    return one_value_fea


# 获得特征的对象类型有哪些，序列类型有哪些，数值类型有哪些，
def get_feature_type(ds: pd.DataFrame):
    cat_fea = list(ds.select_dtypes(exclude=np.number))
    num_fea = list(
        filter(lambda x: x not in cat_fea, list(ds.columns)))

    return cat_fea, num_fea


# 划分数值型变量中的连续变量和离散型变量, 判定数值多于100，认为是连续值，小于100认为是离散值
def get_num_fea_type(ds: pd.DataFrame, num_fea, ordinal_fea=None):
    num_fea = []
    cat_fea = []
    if ordinal_fea is not None:
        return [_ for _ in num_fea if _ not in ordinal_fea], ordinal_fea

    for fea in numerical_fea:
        temp = ds[fea].nunique()
        if temp <= 100:
            cat_fea.append(fea)
            continue
        num_fea.append(fea)
    return num_fea, cat_fea


########################################################################
# 数据分析 - 第三步
# *分布
########################################################################
# 类别型变量分析-分布
def get_cat_distribution(ds: pd.DataFrame, column_name):
    tmp = ds[column_name].value_counts(dropna=False)
    index = min(len(tmp), 20)
    plt.figure(figsize=(8, 8))
    sns.barplot(tmp[:index])
    plt.show()
    return tmp


# 数值连续型变量分析-分布
# 查看某一个数值型变量的分布，查看变量是否符合正态分布，
# 如果想统一处理一批数据变标准化 必须把这些之前已经正态化的数据提出
def get_cat_distribution(ds: pd.DataFrame, column_name):
    f = pd.melt(ds, value_vars=column_name)
    g = sns.FacetGrid(f,
                      col="variable",
                      col_wrap=2,
                      sharex=False,
                      sharey=False)
    g = g.map(sns.distplot, "value")


# 如果不符合正太分布的变量可以log化后再观察下是否符合正态分布
def get_num_distribution_log(ds: pd.DataFrame, column_name):
    plt.figure(figsize=(16, 12))
    plt.suptitle('Transaction Values Distribution', fontsize=22)
    plt.subplot(221)
    sub_plot_1 = sns.distplot(ds[column_name])
    sub_plot_1.set_title(f"{column_name} Distribuition", fontsize=18)
    sub_plot_1.set_xlabel("")
    sub_plot_1.set_ylabel("Probability", fontsize=15)

    plt.subplot(222)
    sub_plot_2 = sns.distplot(np.log(ds[column_name]))
    sub_plot_2.set_title(f"{column_name} (Log) Distribuition", fontsize=18)
    sub_plot_2.set_xlabel("")
    sub_plot_2.set_ylabel("Probability", fontsize=15)


# 根绝y值不同可视化x离散特征的分布
def get_cat_distribution_by_label(ds: pd.DataFrame, label_name, column):
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


# 根绝y值不同可视化x连续特征的分布
def get_num_distribution_by_label(ds: pd.DataFrame, label_name, column):
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

########################################################################
# 数据分析 - 第四步
# *相关性
########################################################################
def ds_corr(ds: pd.DataFrame):
    ds_corr = ds.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    ds_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
    return ds_corr

def ds_corr_graph(ds: pd.DataFrame):
    fig, axs = plt.subplots(nrows=2, figsize=(20, 20))
    sns.heatmap(ds.corr(), ax=axs[0], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})

    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].tick_params(axis='y', labelsize=14)

    axs[0].set_title('Training Set Correlations', size=15)


    plt.show()

# 检测异常的方法一：均方差
# 在统计学中，如果一个数据分布近似正态，
# 那么大约 68% 的数据值会在均值的一个标准差范围内，
# 大约 95% 会在两个标准差范围内，
# 大约 99.7% 会在三个标准差范围内。
def find_outliers_by_3segama(ds, fea):
    data_std = np.std(ds[fea])
    data_mean = np.mean(ds[fea])
    outliers_cut_off = data_std * 3
    lower_rule = data_mean - outliers_cut_off
    upper_rule = data_mean + outliers_cut_off
    ds[fea + '_outliers'] = ds[fea].apply(
        lambda x: str('异常值') if x > upper_rule or x < lower_rule else '正常值')

    print(ds[fea + '_outliers'].value_counts())
    # print(ds.groupby(fea + '_outliers')[label].sum())
    print('*' * 10)
    return ds


# 删除异常值
def del_outliers(ds, fea):
    ds = ds[ds[fea + '_outliers'] == '正常值']
    ds = ds.drop(fea + '_outliers', axis=1)
    ds = ds.reset_index(drop=True)
    return ds


# 通过除法映射到间隔均匀的分箱中，每个分箱的取值范围都是loanAmnt/1000
def bin_num(ds, fea, num=1000):
    ds[fea + "_bin1"] = np.floor_divide(ds[fea], num)


# 通过对数函数映射到指数宽度分箱
def bin_log(ds, fea):
    ds[fea + "_bin2"] = np.floor(np.log10(ds[fea]))


# 分位数分箱
def bin_log(ds, fea):
    ds[fea + "_bin3"] = pd.qcut(ds[fea], 10, labels=False)


# labelEncode 直接放入树模型中
def label_encode(ds, fea):
    #高维类别特征需要进行转换
    le = LabelEncoder()
    le.fit(list(ds[fea].astype(str).values))
    ds[fea] = le.transform(list(ds[fea].astype(str).values))
    return ds


# 类型转换
def to_dtype(ds, fea, dtype="category"):
    # number bool category string object
    ds[fea] = ds[fea].astype(dtype)
    return ds
