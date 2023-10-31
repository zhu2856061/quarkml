# -*- coding: utf-8 -*-
# @Time   : 2023/7/27 09:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import os
import pandas as pd
from loguru import logger

from quarkml.index.woe_iv import WOEIV
from quarkml.index.psi import PSI
from typing import List, Dict
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


class FeatureIndex(object):

    def __init__(self) -> None:
        self.IVObj = WOEIV()
        self.PSIObj = PSI()

    def iv_index(
        self,
        ds: pd.DataFrame,
        label: str,
        part_column: str = None,
        cat_features: List = None,
        part_values: List = None,
        bins=10,
        distributed_and_multiprocess=-1,
        report_dir='encode',
    ):
        # step0: 划分X y
        X = ds.drop(label, axis=1)
        y = ds[[label]]
        """ 基于IV值 大于 0.02 为有效特征进行选择 """
        woe, iv = self.IVObj.fit(
            X=X,
            y=y,
            part_column=part_column,
            cat_features=cat_features,
            part_values=part_values,
            bins=bins,
            distributed_and_multiprocess=distributed_and_multiprocess,
        )
        logger.info(
            f"=== current_feature number: {len(X.columns)}, selected feature number: {len(selected_fea)} ==="
        )

        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(report_dir + "/woe", exist_ok=True)
        for name, ds1 in woe.items():
            ds1.to_csv(report_dir + "/woe/" + name + "_woe.csv")
        tmp = sorted(iv.items(), key=lambda _: _[1], reverse=True)
        iv = pd.DataFrame(tmp)
        iv.to_csv(report_dir + "/iv.csv")

        return iv

    def psi_index(
        self,
        ds: pd.DataFrame,
        label: str,
        part_column: str,
        cat_features: List = None,
        part_values: List = None,
        bins: int = 10,
        distributed_and_multiprocess=-1,
        report_dir='encode',
    ):
        # step0: 划分X y
        X = ds.drop(label, axis=1)

        psi_detail, psi = self.PSIObj.fit(
            X=X,
            part_column=part_column,
            cat_features=cat_features,
            part_values=part_values,
            bins=bins,
            distributed_and_multiprocess=distributed_and_multiprocess,
        )

        logger.info(
            f"=== current_feature number: {len(X.columns)}, selected feature number: {len(selected_fea)} ==="
        )
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(report_dir + "/psi", exist_ok=True)
        for name, ds1 in psi_detail.items():
            ds1.to_csv(report_dir + "/psi/" + name + "_psi.csv")
        psi.to_csv(report_dir + "/psi.csv")

        return psi
