# -*- coding: utf-8 -*-
# @Time   : 2023/7/27 09:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition

from functools import reduce
import os
import pickle
import numpy as np
import ray
from multiprocessing import Pool
import pandas as pd

from quarkml.utils import transform, get_categorical_numerical_features
from typing import List, Dict, Set
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


class PSI(object):
    """PSI = SUM( (实际占比 - 预期占比）* ln(实际占比 / 预期占比) )
       1. 将变量预期分布（excepted）进行分箱（binning）离散化，统计各个分箱里的样本占比
          分箱可以是等频、等距或其他方式，分箱方式不同，将导致计算结果略微有差异；
       2. 按相同分箱区间，对实际分布（actual）统计各分箱内的样本占比
       3. 计算各分箱内的A - E和Ln(A / E)，计算index = (实际占比 - 预期占比）* ln(实际占比 / 预期占比) 。
       4. 将各分箱的index进行求和，即得到最终的PSI
       0 ~ 0.1    稳定性好
       0.1 ~ 0.25 略不稳定
       > 0.25     不稳定
    """

    def __init__(self):
        pass

    def fit(
        self,
        X: pd.DataFrame,
        part_column: str = None,
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
        """ X : 样本
            part_column : 数据X的区间 (时间)切分维度part_columns , 数据分块后，统计块的两两比较
            categorical_features : 类别特征list
            numerical_features : 数值特征list
            part_values : 与 part_column 同时使用，若part_values 设置，则表明按part_values进行划分数据集，若不设置则会用part_column的每个值
            bins : 数值特征的分桶数
            minimal : 最小替换值
            priori: 初始每个特征的psi， 若有则会采用这个初始的，若没有则采用part_column的值[0]
        """
        if categorical_features is None:
            categorical_features, numerical_features = get_categorical_numerical_features(
                X)

        if candidate_features is not None:
            X, _ = transform(X, candidate_features)
            categorical_features, numerical_features = get_categorical_numerical_features(
                X)

        if job < 0:
            job = os.cpu_count()

        psi = pd.DataFrame()
        psi_detail = {}
        base = {}

        X = X.reset_index(drop=True)
        part = X[part_column]
        all_parts = part_values or sorted(part.unique())
        indexes = [part[part == value].index for value in all_parts]

        if distributed_and_multiprocess == 1:
            _binning_numerical_remote = ray.remote(_distribution_numerical_section)
            _binning_categorical_remote = ray.remote(_distribution_categorical_section)
        elif distributed_and_multiprocess == 2:
            pool = Pool(job)

        futures_list = []
        for col in numerical_features:
            if distributed_and_multiprocess == 1:
                futures = _binning_numerical_remote.remote(
                    X,
                    col,
                    indexes,
                    all_parts,
                    minimal,
                    priori,
                    bins,
                )
            elif distributed_and_multiprocess == 2:
                futures = pool.apply_async(_distribution_numerical_section, (
                        X,
                        col,
                        indexes,
                        all_parts,
                        minimal,
                        priori,
                        bins,
                    ))
            else:
                futures = _distribution_numerical_section(
                        X,
                        col,
                        indexes,
                        all_parts,
                        minimal,
                        priori,
                        bins,
                    )
            futures_list.append(futures)

        for col in categorical_features:
            if distributed_and_multiprocess == 1:
                futures = _binning_categorical_remote.remote(
                    X,
                    col,
                    indexes,
                    all_parts,
                    minimal,
                    priori,
                    bins,
                )

            elif distributed_and_multiprocess == 2:
                futures = pool.apply_async(_distribution_categorical_section, (
                        X,
                        col,
                        indexes,
                        all_parts,
                        minimal,
                        priori,
                        bins,
                    ))
            else:
                futures = _distribution_categorical_section(
                        X,
                        col,
                        indexes,
                        all_parts,
                        minimal,
                        priori,
                        bins,
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
            base[col] = items[1]
            psi_detail[col] = items[0]
            tmp_psi_summary = {"var": col}
            tmp_psi_summary.update(items[0].loc["psi"].to_dict())
            psi = psi.append(tmp_psi_summary, ignore_index=True)

        if priori is not None:
            psi = psi[["base"] + all_parts]
        psi.columns = ["psi_" + str(col) for col in psi.columns]

        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "psi.pkl"), "wb") as f:
            pickle.dump({
                "psi": psi,
                "psi_detail": psi_detail,
                "base": base,
            }, f)

        if candidate_features is not None:
            selected_fea = []
            selected_index = []
            keys = list(psi["psi_var"])
            values = [list(psi[_]) for _ in psi.columns if _ != "psi_var"]

            for i, k in enumerate(keys):
                if "booster_f_" in k:
                    is_saved = True
                    for v in values:
                        if v[i] >= 0.25:
                            is_saved = False
                            break
                    if is_saved:
                        indx = int(k.replace("booster_f_", ""))
                        selected_fea.append(candidate_features[indx])
                        selected_index.append(k)
                else:
                    selected_index.append(k)

            return selected_fea, X[selected_index]

        col_name = psi.columns
        for col in col_name:
            if col == "psi_var":
                continue
            psi = psi[psi[col] < 0.25]

        selected_fea = list(psi['psi_var'])
        return selected_fea, X[selected_fea]


def _distribution_categorical_section(X: pd.DataFrame,
                                      col,
                                      indexes,
                                      all_parts,
                                      minimal,
                                      priori,
                                      bins: List = None):
    all_series = [X[col].loc[idx] for idx in indexes]
    if priori is None:
        tmp_base = all_series[0]
        tmp_base_trans, tmp_values = _distribution_categorical(tmp_base)
        all_series = [tmp_base_trans] + [
            _distribution_categorical(series, tmp_values)[0]
            for series in all_series[1:]
        ]
        all_series = [
            series.value_counts(sort=False).sort_index()
            for series in all_series
        ]
        base = [all_series[0], tmp_values]
        tmp_psi = _psi(all_series, all_parts, minimal)
    else:
        tmp_base_trans, tmp_values = priori[col]
        all_series = [
            _distribution_categorical(series, tmp_values)[0]
            for series in all_series
        ]
        all_series = [tmp_base_trans] + [
            series.value_counts(sort=False).sort_index()
            for series in all_series
        ]
        tmp_psi = _psi(all_series, ["base"] + all_parts, minimal)

    return tmp_psi, base


def _distribution_numerical_section(X: pd.DataFrame,
                                    col,
                                    indexes,
                                    all_parts,
                                    minimal,
                                    priori,
                                    bins: List = None):
    all_series = [X[col].loc[idx] for idx in indexes]
    if priori is None:
        tmp_base = all_series[0]
        tmp_base_trans, tmp_bins = _distribution_numerical(tmp_base, bins)
        all_series = [tmp_base_trans] + [
            _distribution_numerical(series, tmp_bins)[0]
            for series in all_series[1:]
        ]
        all_series = [
            series.value_counts(sort=False).sort_index()
            for series in all_series
        ]
        base = [all_series[0], tmp_bins]
        tmp_psi = _psi(all_series, all_parts, minimal)
    else:
        tmp_base_trans, tmp_bins = priori[col]
        all_series = [
            _distribution_numerical(series, tmp_bins)[0]
            for series in all_series
        ]
        all_series = [tmp_base_trans] + [
            series.value_counts(sort=False).sort_index()
            for series in all_series
        ]
        tmp_psi = _psi(all_series, ["base"] + all_parts, minimal)

    return tmp_psi, base


def _distribution_numerical(series: pd.Series, bins: List = None):
    if isinstance(bins, list):
        bins[0] = float("-inf")
        bins[-1] = float("inf")
        series_cut = pd.cut(series, bins=bins)
    else:
        try:
            _, bins = pd.qcut(series, q=bins, retbins=True, duplicates="drop")
        except IndexError:
            bins = [float("-inf"), float("inf")]
        bins[0] = float("-inf")
        bins[-1] = float("inf")
        series_cut = pd.cut(series, bins=bins)
    series_cut.cat.add_categories("_missing", inplace=True)
    series_cut = series_cut.fillna("_missing")
    return series_cut, list(bins)


def _distribution_categorical(series: pd.Series, values: Set = None):
    if values:
        series_trans = series.map(lambda x: str(x)
                                  if x in values else "_missing")
    else:
        values = series.value_counts(sort=False).sort_index().index.tolist()
        series_trans = series.astype("str").fillna("_missing")
    return series_trans, set(values)


def _psi(all_series: List[pd.Series], names: List, minimal: int = 1):
    for series in all_series:
        if "_missing" not in series.index.tolist():
            series.loc["_missing"] = 0
    has_nan = reduce(lambda a, b: a or b,
                     [_.loc["_missing"] for _ in all_series])
    if not has_nan:
        all_series = [series.drop(labels="_missing") for series in all_series]
    base = all_series[0]
    all_series = [base.replace(0, minimal)] + [
        series.reindex(base.index).replace(0, minimal).fillna(minimal)
        for series in all_series[1:]
    ]
    all_series = [series / series.sum() for series in all_series]
    res = pd.DataFrame({names[i]: all_series[i] for i in range(len(names))})
    psi = {names[0]: 0}
    psi.update({
        names[i + 1]: ((all_series[0] - all_series[i + 1]) *
                       np.log(all_series[0] / all_series[i + 1])).sum()
        for i in range(len(names) - 1)
    })
    res.index = res.index.astype("category").add_categories("psi")
    res.loc["psi"] = psi
    return res
