# -*- coding: utf-8 -*-
# @Time   : 2023/7/27 09:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
import os
import numpy as np
import pandas as pd
import ray
from ray.util.multiprocessing import Pool

import pickle
from typing import List
from quarkml.utils import get_cat_num_features, error_callback

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


class WOEIV(object):
    """基于区间列的计算WOE和IV的模块。当设定需要按基期计算时，每个变量的IV
    < 0.02     预测效果 几乎没有
    0.02 ~ 0.1 预测效果 弱
    0.1 ~ 0.3  预测效果 中等
    0.3 ~ 0.5  预测效果 强
    > 0.5      预测效果 难以置信，需确认
    """

    def __init__(self):
        pass

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        part_column: str = None,  # 区间列：时间分块
        cat_features: List = None,
        part_values: List = None,
        handle_zero="merge",
        bins=10,
        minimum=0.5,
        use_base=True,
        distributed_and_multiprocess=-1,
    ):
        """ handle_zero: 针对 zero 进行合并
            bins: 分桶数，等频分桶
            minimum: 某个桶内都是一类，则另一类防止为0 给予最小值
            part_column: 区间列：时间分块，可以进行区间列分块，然后基于分块后的数据计算出每个分块的woe和iv值
            part_values: 区间的值，若是为None 会拿columns 内的所有值
            report_dir: 会保存woe 和iv 的中间值，pd.DataFrame
        """

        categorical_features, numerical_features = get_cat_num_features(X, cat_features)

        job = os.cpu_count() - 2

        assert handle_zero in ["merge", "minimum"
                               ], "Zero handler should be merge or minimum"
        y = y[y.columns[0]]

        woe = {}
        iv = {}
        # 【注意】这里，可以采用多进程模型 pool 池，也可以采用ray 的多进程和分布式

        if part_column:
            if part_values is None:  # 为None 就按该列所有值进行分块
                part_values = sorted(X[part_column].unique())
            segments = {
                seg: X[X[part_column] == seg].index
                for seg in part_values
            }

            if distributed_and_multiprocess == 1:
                _binning_numerical_remote = ray.remote(
                    _binning_numerical_section)
                _binning_categorical_remote = ray.remote(
                    _binning_categorical_section)
            elif distributed_and_multiprocess == 2:
                pool = Pool(job)

            futures_list = []
            for col in numerical_features:
                if distributed_and_multiprocess == 1:
                    futures = _binning_numerical_remote.remote(
                        X[col],
                        y,
                        segments,
                        use_base,
                        handle_zero,
                        bins,
                        minimum,
                        part_column,
                    )
                elif distributed_and_multiprocess == 2:
                    futures = pool.apply_async(_binning_numerical_section, (
                        X[col],
                        y,
                        segments,
                        use_base,
                        handle_zero,
                        bins,
                        minimum,
                        part_column,
                    ),
                                               error_callback=error_callback)
                else:
                    futures = _binning_numerical_section(
                        X[col],
                        y,
                        segments,
                        use_base,
                        handle_zero,
                        bins,
                        minimum,
                        part_column,
                    )
                futures_list.append(futures)

            for col in categorical_features:
                if distributed_and_multiprocess == 1:
                    futures = _binning_categorical_remote.remote(
                        X[col],
                        y,
                        segments,
                        part_column,
                    )
                elif distributed_and_multiprocess == 2:
                    futures = pool.apply_async(_binning_categorical_section, (
                        X[col],
                        y,
                        segments,
                        part_column,
                    ),
                                               error_callback=error_callback)
                else:
                    futures = _binning_categorical_section(
                        X[col],
                        y,
                        segments,
                        part_column,
                    )
                futures_list.append(futures)

            if distributed_and_multiprocess == 2:
                pool.close()
                pool.join()

            if distributed_and_multiprocess == 1:
                futures_list = [_ for _ in ray.get(futures_list)]
            elif distributed_and_multiprocess == 2:
                futures_list = [_.get() for _ in futures_list]

            for col, items in zip(numerical_features + categorical_features,
                                  futures_list):
                woe[col] = items[0]
                iv[col] = items[1]

        else:
            if distributed_and_multiprocess == 1:
                _binning_numerical_remote = ray.remote(_binning_numerical)
                _binning_categorical_remote = ray.remote(_binning_categorical)
            elif distributed_and_multiprocess == 2:
                pool = Pool(job)

            futures_list = []
            for col in numerical_features:
                if distributed_and_multiprocess == 1:
                    futures = _binning_numerical_remote.remote(
                        X[col],
                        y,
                        None,
                        handle_zero,
                        bins,
                        minimum,
                    )
                elif distributed_and_multiprocess == 2:
                    futures = pool.apply_async(_binning_numerical, (
                        X[col],
                        y,
                        None,
                        handle_zero,
                        bins,
                        minimum,
                    ),
                                               error_callback=error_callback)
                else:
                    futures = _binning_numerical(
                        X[col],
                        y,
                        None,
                        handle_zero,
                        bins,
                        minimum,
                    )
                futures_list.append(futures)

            for col in categorical_features:
                if distributed_and_multiprocess == 1:
                    futures = _binning_categorical_remote.remote(
                        X[col],
                        y,
                    )
                elif distributed_and_multiprocess == 2:
                    futures = pool.apply_async(_binning_categorical, (
                        X[col],
                        y,
                    ),
                                               error_callback=error_callback)
                else:
                    futures = _binning_categorical(
                        X[col],
                        y,
                    )
                futures_list.append(futures)

            if distributed_and_multiprocess == 2:
                pool.close()
                pool.join()

            if distributed_and_multiprocess == 1:
                futures_list = [_ for _ in ray.get(futures_list)]
            elif distributed_and_multiprocess == 2:
                futures_list = [_.get() for _ in futures_list]

            for col, items in zip(numerical_features + categorical_features,
                                  futures_list):

                woe[col] = items[0]
                iv[col] = items[0]["iv"].iloc[0] if items[0] is not None else 0

        if part_column:  # 区间的话，就需要每个区间内的值都是大于要求的，即保留每个区间的最小值
            for col, section_v in iv.items():
                iv[col] = min(list(section_v.values()))

        selected_fea = [k for k, v in iv.items() if v > 0.02]
        return selected_fea, X[selected_fea], woe, iv


def _binning_numerical_section(
    feature: pd.Series,
    label: pd.Series,
    segments,
    use_base,
    handle_zero,
    bins,
    minimum,
    part_column,
):
    tmp_bins = None
    tmp_woe = pd.DataFrame()
    tmp_iv = {}
    for seg in segments:
        if not use_base:
            tmp_bins = None
        tmp_feature = feature.loc[segments[seg]]
        tmp_label = label.loc[segments[seg]]
        tmp_res, tmp_bins = _binning_numerical(tmp_feature, tmp_label,
                                               tmp_bins, handle_zero, bins,
                                               minimum)
        tmp_res[part_column] = seg
        tmp_woe = tmp_woe.append(tmp_res)
        tmp_iv[
            f"iv_{seg}"] = tmp_res["iv"].iloc[0] if tmp_res is not None else 0

    return tmp_woe, tmp_iv


def _binning_categorical_section(
    feature: pd.Series,
    label: pd.Series,
    segments,
    part_column,
):

    tmp_woe = pd.DataFrame()
    tmp_iv = {}
    for seg in segments:
        tmp_feature = feature.loc[segments[seg]]
        tmp_label = label.loc[segments[seg]]
        tmp_res, _ = _binning_categorical(tmp_feature, tmp_label)
        tmp_res[part_column] = seg
        tmp_woe = tmp_woe.append(tmp_res, ignore_index=True)
        tmp_iv[
            f"iv_{seg}"] = tmp_res["iv"].iloc[0] if tmp_res is not None else 0

    return tmp_woe, tmp_iv


def _binning_numerical(
        feature: pd.Series,
        label: pd.Series,
        b: List = None,
        handle_zero: str = 'merge',
        bins: int = 10,  # 分桶个数
        minimum: float = 0.5):

    total_good, total_bad = label.value_counts(sort=False).sort_index()
    if b is None:
        try:
            _, b = pd.qcut(feature, q=bins, duplicates="drop",
                           retbins=True)  # 等频分桶
        except (IndexError, ValueError):
            b = [float("-inf"), float("inf")]
        except TypeError:
            warnings.warn("The feature is unable to cut")
            return None, [float("-inf"), float("inf")]

    b[0] = float("-inf")  # 修改下界为 -inf
    b[-1] = float("inf")  # 修改上界为 inf
    feature_cut = pd.cut(feature, bins=b)  # 等距分桶

    tmp = pd.crosstab(feature_cut, label).reset_index()
    tmp[0] = tmp[0] if 0 in tmp.columns else minimum  # 防止 没有label 就=0.5
    tmp[1] = tmp[1] if 1 in tmp.columns else minimum

    if handle_zero == "merge":
        final_b = [float("-inf")]
        max_row = 0
        for row in range(tmp.shape[0]):
            if tmp.iloc[row, 1] and tmp.iloc[row, 2]:
                final_b.append(tmp.iloc[row, 0].right)
                max_row = row
        if max_row < tmp.shape[0] - 1:
            final_b[-1] = float("inf")
        feature_cut = pd.cut(feature, bins=final_b)
        feature_cut.cat.add_categories("_missing", inplace=True)
        feature_cut = feature_cut.fillna("_missing")
        tmp = pd.crosstab(feature_cut, label).reset_index()
        tmp.replace(to_replace=0, value=minimum, inplace=True)
        if not (tmp.iloc[-1, 1] and tmp.iloc[-1, 2]):
            tmp = tmp.iloc[:-1, :]
    else:
        tmp.replace(to_replace=0, value=minimum, inplace=True)
        final_b = b
    if tmp.empty:
        return None
    tmp.columns = ["bin", "good", "bad"]
    tmp["inner_good_ratio"] = tmp["good"] / (tmp["good"] + tmp["bad"])
    tmp["inner_bad_ratio"] = tmp["bad"] / (tmp["good"] + tmp["bad"])

    tmp["global_good_ratio"] = tmp["good"] / total_good
    tmp["global_bad_ratio"] = tmp["bad"] / total_bad
    tmp["woe"] = np.log(tmp["global_bad_ratio"] / tmp["global_good_ratio"])
    tmp["iv_bin"] = (tmp["global_bad_ratio"] -
                     tmp["global_good_ratio"]) * tmp["woe"]
    tmp["iv"] = tmp["iv_bin"].sum()
    return tmp, final_b


def _binning_categorical(
    feature: pd.Series,
    label: pd.Series,
):
    total_good, total_bad = label.value_counts(sort=False).sort_index()
    feature_ = feature.astype("str").fillna("_missing")
    tmp = pd.crosstab(feature_, label).reset_index()
    tmp.columns = ["bin", "good", "bad"]
    tmp["inner_good_ratio"] = tmp["good"] / (tmp["good"] + tmp["bad"])
    tmp["inner_bad_ratio"] = tmp["bad"] / (tmp["good"] + tmp["bad"])
    tmp1 = tmp[(tmp["good"] > 0) & (tmp["bad"] > 0)]
    id_best = tmp1["inner_good_ratio"].idxmax()
    id_worst = tmp1["inner_bad_ratio"].idxmax()
    tmp2_0 = tmp[tmp["good"] == 0]
    tmp2_1 = tmp[tmp["bad"] == 0]
    to_merge = 0

    if not tmp2_0.empty:
        values = ", ".join(tmp2_0["bin"].astype("str").tolist())
        sum_bad = tmp2_0["bad"].sum()
        tmp1.loc[id_worst, "bad"] += sum_bad
        tmp1.loc[id_worst, "bin"] += ", " + values
        to_merge = 1
    if not tmp2_1.empty:
        values = ", ".join(tmp2_1["bin"].astype("str").tolist())
        sum_good = tmp2_1["good"].sum()
        tmp1.loc[id_best, "good"] += sum_good
        tmp1.loc[id_best, "bin"] += ", " + values
        to_merge = 1
    if to_merge:
        tmp1 = tmp1.drop(columns=["inner_good_ratio", "inner_bad_ratio"])
        tmp1["inner_good_ratio"] = tmp1["good"] / (tmp1["good"] + tmp1["bad"])
        tmp1["inner_bad_ratio"] = tmp1["bad"] / (tmp1["good"] + tmp1["bad"])
    tmp1["global_good_ratio"] = tmp1["good"] / total_good
    tmp1["global_bad_ratio"] = tmp1["bad"] / total_bad
    tmp1["woe"] = np.log(tmp1["global_bad_ratio"] / tmp1["global_good_ratio"])
    tmp1["iv_bin"] = (tmp1["global_bad_ratio"] -
                      tmp1["global_good_ratio"]) * tmp1["woe"]
    tmp1["iv"] = tmp1["iv_bin"].sum()
    return tmp1, -1
