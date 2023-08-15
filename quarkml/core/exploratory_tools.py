# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
import pandas_profiling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib
import warnings

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

# 数据报告
def get_report(data_train, dir):
    pfr = pandas_profiling.ProfileReport(data_train)
    pfr.to_file(dir + "/report.html")


# 透视图
def pivot_table(data_train, index, columns, values, aggfunc):
    pivot = pd.pivot_table(data_train,
                           index=index,
                           columns=columns,
                           values=values,
                           aggfunc=aggfunc)
    return pivot


# 查看数据集的样本个数和原始特征维度
def get_shape(ds):
    return ds.shape


# 获取列名
def get_columns(ds):
    return ds.columns


# 数据类型信息
def get_info(ds):
    return ds.info()


# 总体粗略的查看数据集各个特征的一些基本统计量
def get_describe(ds):
    return ds.describe()


#查看数据集中有特征缺失值的列数
def get_isnull_column_num(ds):
    return ds.isnull().any().sum()


# 上面得到训练集有22列特征有缺失值，进一步查看特征中缺失率
def get_column_missing_rate(ds):
    missing = (ds.isnull().sum() / len(ds))
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    missing.plot.bar()
    return missing.to_dict()


# 查看哪些特征只有一个值的
def get_one_value_feature(ds):
    one_value_fea = [col for col in ds.columns if ds[col].nunique() <= 1]
    return one_value_fea


# 获得特征的对象类型有哪些，序列类型有哪些，数值类型有哪些，
def get_feature_type(ds: pd.DataFrame):
    category_fea = list(ds.select_dtypes(exclude=np.number))
    numerical_fea = list(filter(lambda x: x not in category_fea, list(ds.columns)))

    return category_fea, numerical_fea,


# 划分数值型变量中的连续变量和离散型变量, 判定数值多于100，认为是连续值，小于100认为是离散值
def get_numerical_serial_fea(ds, numerical_fea, ordinal_fea=None):
    numerical_serial_fea = []
    numerical_noserial_fea = []
    if ordinal_fea is not None:
        return [_ for _ in numerical_fea if _ not in ordinal_fea], ordinal_fea

    for fea in numerical_fea:
        temp = ds[fea].nunique()
        if temp <= 100:
            numerical_noserial_fea.append(fea)
            continue
        numerical_serial_fea.append(fea)
    return numerical_serial_fea, numerical_noserial_fea


# 数值类别型变量分析-分布
def get_numerical_noserial_distribution(ds, column_name):
    tmp = ds[column_name].value_counts(dropna=False)
    index = min(len(tmp), 20)
    plt.figure(figsize=(8, 8))
    sns.barplot(tmp[:index])
    plt.show()
    return tmp


# 数值连续型变量分析-分布
# 查看某一个数值型变量的分布，查看变量是否符合正态分布，
# 如果想统一处理一批数据变标准化 必须把这些之前已经正态化的数据提出
def get_numerical_serial_distribution(ds, numerical_serial_fea):
    f = pd.melt(ds, value_vars=numerical_serial_fea)
    g = sns.FacetGrid(f,
                      col="variable",
                      col_wrap=2,
                      sharex=False,
                      sharey=False)
    g = g.map(sns.distplot, "value")


# 如果不符合正太分布的变量可以log化后再观察下是否符合正态分布
def get__numerical_serial_distribution_log(ds, column_name):
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
def get_category_distribution_by_label(ds, label_name, column):
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
def get_numerical_serial_distribution_by_label(ds, label_name, column):
    label_values = ds[label_name].unique()
    fig, ax = plt.subplots(len(label_values), 1, figsize=(15, 20))
    for i, lv in enumerate(label_values):
        train_loan_fr = ds[[label_name, column]].loc[ds[label_name] == lv]
        train_loan_fr[column].plot(
            kind='hist', ax=ax[i], color='b', bins=100, title=f'Count of {column} label = {lv}')
    plt.savefig(f'encode/{column}.png')


# 把所有缺失值替换为指定的值0
def fillna(ds, numerical_fea, category_fea):
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

# 对类型进行压缩
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df