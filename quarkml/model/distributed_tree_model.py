# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import numpy as np
from loguru import logger
import pandas as pd
import time
from sklearn.metrics import roc_auc_score, roc_curve
from ray.train.lightgbm import LightGBMTrainer, LightGBMPredictor
from ray.data.preprocessors.chain import Chain
from ray.data.preprocessors.encoder import Categorizer
from ray.data.preprocessors import StandardScaler
from ray.data import ActorPoolStrategy
from ray.air.config import ScalingConfig, RunConfig, CheckpointConfig
from typing import List
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


def lgb_distributed_train(
    trn_ds,
    label_name,
    val_ds=None,
    categorical_features: List = None,
    params=None,
    seed=2023,
    num_workers=2,
    trainer_resources=None,
    resources_per_worker=None,
    use_gpu=False,
    report_dir = './encode/dist_model',
):

    start = time.time()
    auc_score = None
    ks_score = None

    if val_ds is None:
        trn_ds, val_ds = trn_ds.train_test_split(
            test_size=0.3,
            seed=seed,
        )

    # defualt config
    params_set = {
        'objective': 'regression',
        'n_estimators': 2000,
        'boosting_type': 'gbdt',
        'importance_type': 'gain',
        'metric': 'auc',
        'num_leaves': 2**3,
        'max_depth': 3,
        'min_data_in_leaf': 5,
        'min_gain_to_split': 0,
        'lambda_l1': 2,
        'lambda_l2': 2,
        'bagging_freq': 3,
        'bagging_fraction': 0.7,
        'learning_rate': 0.1,
        'seed': seed,
        'feature_pre_filter': False,
        'verbosity': -1,
        'period': 100,
        'early_stopping_round': 200,
    }

    _label_unique_num = trn_ds.unique(label_name)
    if len(_label_unique_num) <= 2:
        params_set.update({'objective': 'regression'})
        params_set.update({'metric': 'auc'})
    else:
        params_set.update({'objective': 'multiclass'})
        params_set.update({'num_class': len(_label_unique_num)})
        params_set.update({'metric': 'auc_mu'})

    if params is not None:
        params_set.update(params)

    logger.info(f"************************************ model parameters : {params_set} ************************************")

    # Scale some random columns, and categorify the categorical_column,
    # allowing LightGBM to use its built-in categorical feature support
    preprocessor = None
    if categorical_features is not None and len(categorical_features) > 0:
        preprocessor = Chain(
            Categorizer(categorical_features), 
        )
    
    ''' 【注意】
    1 多机多gpu / 或者单机多gpu-> 
    scaling_config = ScalingConfig(
    num_workers=2, # 几个GPU就设置几
    use_gpu=True,
    )

    2 多机多cpu ->
    scaling_config = ScalingConfig(
    num_workers=4,
    trainer_resources={"CPU": 0},
    resources_per_worker={"CPU": 8},
    )

    3 若是单机cpu 就没必要分布式了，并发lgb和xgb 都具备了
    '''
    gbm = LightGBMTrainer(
        scaling_config=ScalingConfig(
            # Number of workers to use for data parallelism.
            num_workers=num_workers,
            trainer_resources=trainer_resources,
            resources_per_worker=resources_per_worker,
            # Whether to use GPU acceleration.d
            use_gpu=use_gpu,
        ),
        run_config=RunConfig(
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=1,
                num_to_keep=1,
                # checkpoint_score_attribute='valid-auc',
                # checkpoint_score_order='min',
            ),
            storage_path = report_dir,
        ),
        preprocessor=preprocessor,
        label_column=label_name,
        num_boost_round=params_set['n_estimators'],
        params=params_set,
        datasets={"train": trn_ds, "valid": val_ds},
    )
    result = gbm.fit()

    # 评估
    report_dict = None
    if len(_label_unique_num) <= 2:
        y_true = val_ds.select_columns(cols=[label_name]).to_pandas()[label_name]

        test_dataset = val_ds.drop_columns(cols=[label_name])
        y_scores = test_dataset.map_batches(
            DistributedLGBPredict, 
            fn_constructor_args=[result.checkpoint], 
            compute=ActorPoolStrategy(), 
            batch_format="pandas"
        )
        y_scores = y_scores.to_pandas()['predictions']

        if params_set['objective'] == 'regression':
            auc_score = _auc(y_true, y_scores)
            ks_score = _ks(y_true, y_scores)
            logger.info(f"lgb_model: auc:{auc_score}, ks:{ks_score}")

        report_dict = {
            'auc_score': auc_score,
            'ks_score': ks_score,
        }
    logger.info(
        f'************************************ end lgb total time: {time.time()-start} ************************************'
    )

    return gbm, report_dict, params_set


class DistributedLGBPredict(object):
    def __init__(self, checkpoint) -> None:
        self.predictor = LightGBMPredictor.from_checkpoint(checkpoint)

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        return self.predictor.predict(batch)


def _auc(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    return roc_auc_score(y_true, y_scores)


def _ks(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    FPR, TPR, thresholds = roc_curve(y_true, y_scores)
    return abs(FPR - TPR).max()
