# -*- coding: utf-8 -*-
# @Time   : 2023/7/27 09:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
from loguru import logger
from copy import deepcopy
from typing import List
from quarkml.utils import (
    Node,
    FNode,
    tree_to_formula,
    get_cat_num_features,
)

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)

all_operators = ["freq"]
num_operators = [
    "abs", "log", "sqrt", "square", "sigmoid", "round", "residual"
]
num_num_operators = ["min", "max", "+", "-", "*", "/"]
cat_num_operators = [
    "GroupByThenMin", "GroupByThenMax", "GroupByThenMean", "GroupByThenMedian",
    "GroupByThenStd", "GroupByThenRank"
]
cat_cat_operators = ["Combine", "CombineThenFreq", "GroupByThenNUnique"]

symmetry_operators = [
    "min", "max", "+", "-", "*", "/", "Combine", "CombineThenFreq"
]
cal_all_operators = [
    "freq", "GroupByThenMin", "GroupByThenMax", "GroupByThenMean",
    "GroupByThenMedian", "GroupByThenStd", "GroupByThenRank", "Combine",
    "CombineThenFreq", "GroupByThenNUnique"
]


class BasicGeneration(object):

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, categorical_features: List = None):

        cat_features, num_features = get_cat_num_features(
            X, categorical_features)
        cat_features.sort()
        num_features.sort()
        logger.info(f"===== 【categorical_features】: {cat_features} =====")
        logger.info(f"===== 【numerical_features】: {num_features} =====")

        candidate_features = self._get_candidate_features(
            cat_features,
            num_features,
        )

        candidate_features = [tree_to_formula(_) for _ in candidate_features]
        candidate_features.sort()

        return candidate_features

    def _get_candidate_features(
        self,
        cat_features: List = [],
        num_features: List = [],
        order=1,
    ):
        # 根据给的数据特征列表获得候选特征集合
        assert len(set(num_features) & set(cat_features)) == 0

        current_order_num_features = []
        current_order_cat_features = []
        for f in num_features + cat_features:
            if f in cat_features:
                current_order_cat_features.append(FNode(f))
            else:
                current_order_num_features.append(FNode(f))

        lower_order_num_features = []
        lower_order_cat_features = []
        candidate_features_list = []

        while order > 0:
            _num, _cat = self._enumerate(current_order_num_features,
                                         lower_order_num_features,
                                         current_order_cat_features,
                                         lower_order_cat_features)
            candidate_features_list.extend(_num)
            candidate_features_list.extend(_cat)
            lower_order_num_features, lower_order_cat_features = current_order_num_features, current_order_cat_features
            current_order_num_features, current_order_cat_features = _num, _cat
            order -= 1
        return candidate_features_list

    def _enumerate(self, current_order_num_features, lower_order_num_features,
                   current_order_cat_features, lower_order_cat_features):
        num_candidate_features = []
        cat_candidate_features = []
        for op in all_operators:
            for f in current_order_num_features + current_order_cat_features:
                num_candidate_features.append(Node(op, children=[deepcopy(f)]))
        for op in num_operators:
            for f in current_order_num_features:
                num_candidate_features.append(Node(op, children=[deepcopy(f)]))
        for op in num_num_operators:
            for i in range(len(current_order_num_features)):
                f1 = current_order_num_features[i]
                k = i if op in symmetry_operators else 0
                for f2 in current_order_num_features[
                        k:] + lower_order_num_features:
                    if self._check_xor(f1, f2):
                        num_candidate_features.append(
                            Node(op, children=[deepcopy(f1),
                                               deepcopy(f2)]))

        for op in cat_num_operators:
            for f in current_order_num_features:
                for cat_f in current_order_cat_features + lower_order_cat_features:
                    if self._check_xor(f, cat_f):
                        num_candidate_features.append(
                            Node(op, children=[deepcopy(f),
                                               deepcopy(cat_f)]))
            for f in lower_order_num_features:
                for cat_f in current_order_cat_features:
                    if self._check_xor(f, cat_f):
                        num_candidate_features.append(
                            Node(op, children=[deepcopy(f),
                                               deepcopy(cat_f)]))

        for op in cat_cat_operators:
            for i in range(len(current_order_cat_features)):
                f1 = current_order_cat_features[i]
                k = i if op in symmetry_operators else 0
                for f2 in current_order_cat_features[
                        k:] + lower_order_cat_features:
                    if self._check_xor(f1, f2):
                        if op in ['Combine']:
                            cat_candidate_features.append(
                                Node(op, children=[deepcopy(f1),
                                                   deepcopy(f2)]))
                        else:
                            num_candidate_features.append(
                                Node(op, children=[deepcopy(f1),
                                                   deepcopy(f2)]))
        return num_candidate_features, cat_candidate_features

    def _check_xor(self, node1, node2):

        def _get_FNode(node):
            if isinstance(node, FNode):
                return [node.name]
            else:
                res = []
                for child in node.children:
                    res.extend(_get_FNode(child))
                return res

        fnode1 = set(_get_FNode(node1))
        fnode2 = set(_get_FNode(node2))
        if len(fnode1 ^ fnode2) == 0:
            return False
        else:
            return True

    def _get_categorical_numerical_features(self, ds: pd.DataFrame):
        # 获取类别特征，除number类型外的都是类别特征
        categorical_features = list(ds.select_dtypes(exclude=np.number))
        numerical_features = []
        for feature in ds.columns:
            if feature in categorical_features:
                continue
            else:
                numerical_features.append(feature)

        return categorical_features, numerical_features
