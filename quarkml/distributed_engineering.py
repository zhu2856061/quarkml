# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

from loguru import logger
import pandas as pd

from quarkml.model.tree_model import lgb_train
from quarkml.feature_engineering import FeatureEngineering
FE = FeatureEngineering()

import ray
context = ray.init()
print(context.dashboard_url)

# step1
ds = pd.read_csv("../experiment/credit/credit-g.arff")
ds, cat, con = FE.data_processing(ds, 'class', is_fillna=True, verbosity=False)
print("-1->", ds)
# # step1.1
# testds = pd.read_csv("../experiment/credit/credit-g.arff")
# ds = FE.data_processing(testds, 'class', task='tranform', verbosity=False)
# print("-2->", ds)
# step2
X = ds.drop('class', axis=1)
y = ds[['class']]

def ray_lgb(
    trn_x: pd.DataFrame,
    trn_y: pd.DataFrame,
):
    return lgb_train(
        trn_x, trn_y
    )

train_remote = ray.remote(ray_lgb)
futures = [train_remote.remote(X, y) for i in range(5)]

for _ in ray.get(futures):
    print(_)