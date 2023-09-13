# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

from quarkml.core.distributed_data_processing import DistDataProcessing
from quarkml.core.feature_selector import FeatureSelector
from quarkml.core.model_train import TreeModel
from typing import List
import pandas as pd


class DistributedEngineering(object):

    def __init__(self) -> None:
        self.DDP = DistDataProcessing()
        self.FS = FeatureSelector()

        self.TM = TreeModel()

    def dist_data_processing(
        self,
        files=None,
        label_name: str = None,
        delimiter: str = ',',
        cat_feature: List = [],
        num_feature: List = [],
        ordinal_number=100,
        report_dir="./encode",
        ds: pd.DataFrame = None,
    ):

        if ds is not None:
            return self.DDP.from_pandas(ds, cat_feature)

        return self.DDP.fit(
            files,
            label_name,
            delimiter,
            cat_feature,
            num_feature,
            ordinal_number,
            report_dir,
        )

    def dist_feature_selector(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        candidate_features: List[str] = None,
        categorical_features: List[str] = None,
        numerical_features: List[str] = None,
        params: dict = None,
        tmodel_method='mean',
        init_score: pd.DataFrame = None,
        importance_metric: str = 'importance',
        select_method='predictive',
        min_candidate_features=200,
        blocks=2,
        ratio=0.5,
        folds=5,
        seed=2023,
        part_column: str = None,
        part_values: List = None,
        handle_zero="merge",
        bins=10,
        minimum=0.5,
        minimal: int = 1,
        priori=None,
        use_base=True,
        report_dir="encode",
        method: str = "booster",
        job=-1,
    ):
        distributed_and_multiprocess = 1,  # 采用分布式

        assert method not in [
            "booster", "iv", "psi", "tmodel"
        ], "distributed method must in [booster, iv, psi, tmodel]"

        if method == "booster":
            selected_feature, new_X = self.FS.booster_selector(
                X,
                y,
                candidate_features,
                categorical_features,
                params,
                select_method,
                min_candidate_features,
                blocks,
                ratio,
                seed,
                report_dir,
                distributed_and_multiprocess,
                job,
            )

        if method == "iv":
            selected_feature, new_X = self.FS.iv_selector(
                X,
                y,
                candidate_features,
                categorical_features,
                numerical_features,
                part_column,
                part_values,
                handle_zero,
                bins,
                minimum,
                use_base,
                report_dir,
                distributed_and_multiprocess,
                job,
            )

        if method == "psi":
            selected_feature, new_X = self.FS.psi_selector(
                X,
                part_column,
                candidate_features,
                categorical_features,
                numerical_features,
                part_values,
                bins,
                minimal,
                priori,
                report_dir,
                distributed_and_multiprocess,
                job,
            )

        if method == "tmodel":
            selected_feature, new_X = self.FS.tmodel_selector(
                X,
                y,
                candidate_features,
                categorical_features,
                init_score,
                tmodel_method,
                params,
                importance_metric,
                folds,
                seed,
                report_dir,
                distributed_and_multiprocess,
                job,
            )

        return selected_feature, new_X

    def dist_model(
        self,
        trn_ds,
        label_name,
        val_ds=None,
        categorical_features = None,
        params=None,
        seed=2023,
        num_workers=2,
        trainer_resources=None,
        resources_per_worker=None,
        use_gpu=False,
        report_dir = './encode/dist_model',
    ):
        tmodel = self.TM.lgb_distributed_model(
                trn_ds,
                label_name,
                val_ds,
                categorical_features,
                params,
                seed,
                num_workers,
                trainer_resources,
                resources_per_worker,
                use_gpu,
                report_dir,
        )

    def dist_model_cv(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        categorical_features=None,
        params=None,
        folds=5,
        seed=2023,
        job = -1,
    ):
        distributed_and_multiprocess=1,
        return self.TM.lgb_model_cv(
            X_train,
            y_train,
            X_test,
            y_test,
            categorical_features,
            params,
            folds,
            seed,
            distributed_and_multiprocess,
            job,
        )
