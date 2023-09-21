# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

from quarkml.core.distributed_data_processing import DistDataProcessing
from quarkml.core.model_train import TreeModel
from typing import List
import pandas as pd
from pyarrow import csv
from ray import data as rdata
from loguru import logger
import time


class DistributedEngineering(object):

    def __init__(self) -> None:
        self.DDP = DistDataProcessing()
        self.TM = TreeModel()

    def dist_data_processing(
        self,
        ds: pd.DataFrame,
        label: str = None,
        delimiter: str = ',',
        cat_feature: List = [],
        num_feature: List = [],
        ordinal_number=100,
        report_dir="./encode",
    ):

        start = time.time()
        if isinstance(ds, str):
            ds = self.DDP.fit(
                ds,
                label,
                delimiter,
                cat_feature,
                num_feature,
                ordinal_number,
                report_dir,
            )
        else:
            ds = self.DDP.from_pandas(ds, cat_feature)
        logger.info(
            f'*************** [hparams] cost time: {time.time()-start} ***************'
        )
        return ds

    def dist_model(
        self,
        ds: rdata.Dataset,
        label: str,
        valid_ds: rdata.Dataset = None,
        cat_features=None,
        params=None,
        num_workers=2,
        trainer_resources=None,
        resources_per_worker=None,
        use_gpu=False,
        delimiter=",",
        report_dir='./encode/dist_model',
    ):
        start = time.time()
        if isinstance(ds, str):
            parse_options = csv.ParseOptions(delimiter=delimiter)
            ds = rdata.read_csv(ds, parse_options=parse_options)

        if valid_ds is not None:
            if isinstance(valid_ds, str):
                parse_options = csv.ParseOptions(delimiter=delimiter)
                valid_ds = rdata.read_csv(valid_ds, parse_options=parse_options)

        gbm, result = self.TM.lgb_distributed_model(
            trn_ds=ds,
            label=label,
            valid_ds=valid_ds,
            cat_features=cat_features,
            params=params,
            num_workers=num_workers,
            trainer_resources=trainer_resources,
            resources_per_worker=resources_per_worker,
            use_gpu=use_gpu,
            report_dir=report_dir,
        )

        logger.info(
            f'*************** [hparams] cost time: {time.time()-start} ***************'
        )

        return gbm, result
