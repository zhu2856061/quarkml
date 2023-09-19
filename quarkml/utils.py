# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import os
import sys
import pandas as pd
import numpy as np
from loguru import logger

import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


# 组合节点
class Node(object):

    def __init__(self, op, children):
        self.name = op
        self.children = children
        self.score = 0
        self.data = None
        self.train_idx = []
        self.val_idx = []

    def set_score(self, score):
        self.score = score

    def get_score(self):
        return self.score

    def get_fnode(self):
        fnode_list = []
        for child in self.children:
            fnode_list.extend(child.get_fnode())
        return list(set(fnode_list))

    def delete(self):
        self.data = None
        for child in self.children:
            child.delete()

    def f_delete(self):
        for child in self.children:
            child.f_delete()

    def calculate(self, data, is_root=False):

        if self.name == "abs":
            d = self.children[0].calculate(data)
            new_data = d.abs()
        elif self.name == "log":
            d = self.children[0].calculate(data)
            new_data = np.log(np.abs(d.replace(0, np.nan)))
        elif self.name == "sqrt":
            d = self.children[0].calculate(data)
            new_data = np.sqrt(np.abs(d))
        elif self.name == "square":
            d = self.children[0].calculate(data)
            new_data = np.square(d)
        elif self.name == "sigmoid":
            d = self.children[0].calculate(data)
            new_data = 1 / (1 + np.exp(-d))
        elif self.name == "freq":
            d = self.children[0].calculate(data)
            value_counts = d.value_counts()
            value_counts.loc[None] = None
            value_counts.loc[np.nan] = np.nan
            new_data = d.apply(lambda x: value_counts.loc[x])
        elif self.name == "round":
            d = self.children[0].calculate(data)
            new_data = np.floor(d)
        elif self.name == "residual":
            d = self.children[0].calculate(data)
            new_data = d - np.floor(d)

        elif self.name == "max":
            d1 = self.children[0].calculate(data)
            d2 = self.children[1].calculate(data)
            new_data = np.maximum(d1, d2)
        elif self.name == "min":
            d1 = self.children[0].calculate(data)
            d2 = self.children[1].calculate(data)
            new_data = np.minimum(d1, d2)
        elif self.name == "+":
            d1 = self.children[0].calculate(data)
            d2 = self.children[1].calculate(data)
            new_data = d1 + d2
        elif self.name == "-":
            d1 = self.children[0].calculate(data)
            d2 = self.children[1].calculate(data)
            new_data = d1 - d2
        elif self.name == "*":
            d1 = self.children[0].calculate(data)
            d2 = self.children[1].calculate(data)
            new_data = d1 * d2
        elif self.name == "/":
            d1 = self.children[0].calculate(data)
            d2 = self.children[1].calculate(data)
            new_data = d1 / d2.replace(0, np.nan)
        else:
            d1 = self.children[0].calculate(data)
            d2 = self.children[1].calculate(data)

            if self.name == "GroupByThenMin":
                temp = d1.groupby(d2).min()
                temp.loc[np.nan] = np.nan
                temp.loc[None] = None
                new_data = d2.apply(lambda x: temp.loc[x])
            elif self.name == "GroupByThenMax":
                temp = d1.groupby(d2).max()
                temp.loc[None] = None
                temp.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: temp.loc[x])
            elif self.name == "GroupByThenMean":
                temp = d1.groupby(d2).mean()
                temp.loc[None] = None
                temp.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: temp.loc[x])
            elif self.name == "GroupByThenMedian":
                temp = d1.groupby(d2).median()
                temp.loc[None] = None
                temp.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: temp.loc[x])
            elif self.name == "GroupByThenStd":
                temp = d1.groupby(d2).std()
                temp.loc[None] = None
                temp.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: temp.loc[x])
            elif self.name == 'GroupByThenRank':
                new_data = d1.groupby(d2).rank(ascending=True, pct=True)
            elif self.name == "GroupByThenFreq":

                def _f(x):
                    value_counts = x.value_counts()
                    value_counts.loc[None] = None
                    value_counts.loc[np.nan] = np.nan
                    return x.apply(lambda x: value_counts.loc[x])

                new_data = d1.groupby(d2).apply(_f)
            elif self.name == "GroupByThenNUnique":
                nunique = d1.groupby(d2).nunique()
                nunique.loc[None] = None
                nunique.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: nunique.loc[x])
            elif self.name == "Combine":
                temp = d1.astype(str) + '_' + d2.astype(str)
                temp[d1.isna() | d2.isna()] = np.nan
                temp, _ = temp.factorize()
                new_data = pd.Series(temp, index=d1.index).astype("float64")
            elif self.name == "CombineThenFreq":
                temp = d1.astype(str) + '_' + d2.astype(str)
                temp[d1.isna() | d2.isna()] = np.nan
                value_counts = temp.value_counts()
                value_counts.loc[np.nan] = np.nan
                value_counts.loc[None] = None
                new_data = temp.apply(lambda x: value_counts.loc[x])
            else:
                raise NotImplementedError(
                    f"Unrecognized operator {self.name}.")
        if self.name == 'Combine':
            new_data = new_data.astype('category')
        else:
            new_data = new_data.astype('float')
        if is_root:
            self.data = new_data
        return new_data


# 原始节点
class FNode(object):

    def __init__(self, name):
        self.name = name
        self.data = None
        self.calculate_all = False

    def delete(self):
        self.data = None

    def f_delete(self):
        self.data = None

    def get_fnode(self):
        return [self.name]

    def calculate(self, data):
        self.data = data[self.name]
        return self.data


# 转换成形式化输出
def tree_to_formula(tree):
    if isinstance(tree, Node):
        if tree.name in ['+', '-', '*', '/']:
            string_1 = tree_to_formula(tree.children[0])
            string_2 = tree_to_formula(tree.children[1])
            return str('(' + string_1 + tree.name + string_2 + ')')
        else:
            result = [tree.name, '(']
            for i in range(len(tree.children)):
                string_i = tree_to_formula(tree.children[i])
                result.append(string_i)
                result.append(',')
            result.pop()
            result.append(')')
            return ''.join(result)
    elif isinstance(tree, FNode):
        return str(tree.name)
    else:
        return str(tree.name)


# 形式化输出转换成Node
def formula_to_tree(string):
    if string[-1] != ')':
        return FNode(string)

    def is_trivial_char(c):
        return not (c in '()+-*/,')

    def find_prev(string):
        if string[-1] != ')':
            return max([(0 if is_trivial_char(c) else i + 1)
                        for i, c in enumerate(string)])
        level, pos = 0, -1
        for i in range(len(string) - 1, -1, -1):
            if string[i] == ')': level += 1
            if string[i] == '(': level -= 1
            if level == 0:
                pos = i
                break
        while (pos > 0) and is_trivial_char(string[pos - 1]):
            pos -= 1
        return pos

    p2 = find_prev(string[:-1])
    if string[p2 - 1] == '(':
        return Node(string[:p2 - 1], [formula_to_tree(string[p2:-1])])
    p1 = find_prev(string[:p2 - 1])
    if string[0] == '(':
        return Node(string[p2 - 1], [
            formula_to_tree(string[p1:p2 - 1]),
            formula_to_tree(string[p2:-1])
        ])
    else:
        return Node(string[:p1 - 1], [
            formula_to_tree(string[p1:p2 - 1]),
            formula_to_tree(string[p2:-1])
        ])


def node_to_file(path):
    text = open(path, 'r').read().split('\n')
    res = []
    for s in text:

        if len(s) == 0 or s[-1] != ')': continue
        res.append(formula_to_tree(s))
    return res


def file_to_node(path):
    text = open(path, 'r').read().split('\n')
    res = []
    for s in text:

        if len(s) == 0 or s[-1] != ')': continue
        res.append(formula_to_tree(s))
    return res


def transform(data: pd.DataFrame,
              new_features_list: List[str] = None,
              tmp_save_path: str = './booster_tmp_data.feather',
              report_dir="encode",
              n_jobs=4):

    if new_features_list is None:
        new_features_list = from_csv(report_dir)
    else:
        new_features_list = [formula_to_tree(fea) for fea in new_features_list]
    if new_features_list is None:
        return data
    data.index.name = 'openfe_index'
    data.reset_index().to_feather(tmp_save_path)

    logger.info("Start transforming data.")
    start = datetime.now()
    ex = ThreadPoolExecutor(n_jobs)
    results = []
    for feature in new_features_list:
        results.append(ex.submit(trans, feature))
    ex.shutdown(wait=True)
    logger.info(f"Time spent calculating new features {datetime.now()-start}.")
    _data = []
    names = []
    names_map = {}
    cat_feats = []
    for i, res in enumerate(results):
        is_cat, d1, f = res.result()
        names.append('booster_f_%d' % i)
        names_map['booster_f_%d' % i] = f
        _data.append(d1)
        if is_cat: cat_feats.append('booster_f_%d' % i)
    _data = np.vstack(_data)
    _data = pd.DataFrame(_data.T, columns=names, index=data.index)
    for c in _data.columns:
        if c in cat_feats:
            _data[c] = _data[c].astype('category')
        else:
            _data[c] = _data[c].astype('float')
    _data = pd.concat([data, _data], axis=1)
    logger.info("Finish transformation.")
    os.remove(tmp_save_path)
    return _data, names

def trans(feature: Node, tmp_save_path: str = './booster_tmp_data.feather',):
    try:
        base_features = ['openfe_index']
        base_features.extend(feature.get_fnode())
        _data = pd.read_feather(
            tmp_save_path,
            columns=base_features).set_index('openfe_index')
        feature.calculate(_data, is_root=True)
        if (str(feature.data.dtype) == 'category') | (str(
                feature.data.dtype) == 'object'):
            pass
        else:
            feature.data = feature.data.replace([-np.inf, np.inf], np.nan)
            feature.data = feature.data.fillna(0)
    except:
        print(traceback.format_exc())
        sys.exit()
    return ((str(feature.data.dtype) == 'category')
            or (str(feature.data.dtype) == 'object')
            ), feature.data.values.ravel(), tree_to_formula(feature)

def from_csv(file_dir):
    if os.path.exists(file_dir + "/booster.csv"):
        stage1 = pd.read_csv(file_dir + "/booster.csv")
        res = []
        for i in range(stage1.shape[0]):
            res.append(formula_to_tree(stage1.iloc[i][0]))
        return res
    return None

def to_csv(features_scores, file_dir):
        stage1_dic = {
            'stage1': [],
            'score': [],
        }
        for fea, sc in features_scores:
            stage1_dic['stage1'].append(tree_to_formula(fea))
            stage1_dic['score'].append(sc)

        stage1 = pd.DataFrame(stage1_dic)
        stage1.to_csv(file_dir + "/booster.csv", index=False)

def get_categorical_numerical_features(ds: pd.DataFrame):
    # 获取类别特征，除number类型外的都是类别特征
    categorical_features = list(ds.select_dtypes(exclude=np.number))
    numerical_features = []
    for feature in ds.columns:
        if feature in categorical_features:
            continue
        else:
            numerical_features.append(feature)

    return categorical_features, numerical_features

def error_callback(error):
    logger.info(f"Error info: {error}")