# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function
import sys
sys.path.append("../..")
# ray start --head --port=1063 --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265
import ray
import pandas as pd
from quarkml.feature_engineering import FeatureEngineering
from quarkml.model_engineering import ModelEngineering
from quarkml.distributed_engineering import DistributedEngineering

FE = FeatureEngineering()
DE = DistributedEngineering()
ME = ModelEngineering()
runtime_envs = {"working_dir": "../.."}
context = ray.init(runtime_env = runtime_envs)
print(context.dashboard_url)

# 读取数据
# step1
ds = pd.read_csv("credit.csv")
ds, cat, con = FE.data_processing(ds, 'class', is_fillna=True, verbosity=False)
print("-1->", ds)
tr_ds = ds[:800]
va_ds = ds[800:]

tr_x = tr_ds.drop('class', axis=1)
tr_y = tr_ds[['class']]
va_x = va_ds.drop('class', axis=1)
va_y = va_ds[['class']]

tr_ds = DE.dist_data_processing(ds=tr_ds)
va_ds = DE.dist_data_processing(ds=va_ds)

# print("-1->", tr_ds.schema().types)
# print("-1.1->", tr_ds.count())
# print("-2->", tr_ds.take(4))

# 观察分布式训练和常规训练是否一致
DE.dist_model(tr_ds, 'class', va_ds)
ME.model(tr_x,tr_y, va_x, va_y)

ray.shutdown()