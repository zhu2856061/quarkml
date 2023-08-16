# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

#
# 集群 10.13.26.25 10.13.26.27
#

import ray
import pandas as pd
from quarkml.feature_engineering import FeatureEngineering
from quarkml.model_engineering import ModelEngineering
from quarkml.distributed_engineering import DistributedEngineering
from sklearn.model_selection import train_test_split

FE = FeatureEngineering()
DE = DistributedEngineering()
ME = ModelEngineering()
context = ray.init(address="10.13.26.27:1063")
print(context.dashboard_url)

import time

start = time.time()

# 读取数据
# step1
ds = pd.read_csv(
    "/home/ai_data3/jupyter_workspace/wt_leeweali/adpay_cash_x_model/data/train_data/gcpt_mart_adpay_cash_x_model_sample_v6_20230201.csv",
    sep='\t')

ds, cat, con = FE.data_processing(ds, 'y1', is_fillna=True, verbosity=False)
print("--1--->", time.time() - start)

tr_ds = ds[:int(len(ds) * 0.85)]
va_ds = ds[int(len(ds) * 0.85):]
tr_x = tr_ds.drop('y1', axis=1)
tr_y = tr_ds[['y1']]
va_x = va_ds.drop('y1', axis=1)
va_y = va_ds[['y1']]

tr_ds = DE.dist_data_processing(ds=tr_ds)
va_ds = DE.dist_data_processing(ds=va_ds)

# print("-1.1->", tr_ds.count())

start = time.time()
# 观察分布式训练和常规训练是否一致
DE.dist_model(
    tr_ds,
    'y1',
    va_ds,
    params={'device_type': 'gpu'},
    num_workers=4,
    trainer_resources=None,
    resources_per_worker=None,
    use_gpu=True,
    report_dir='~/dist_model',
)

print("--2--->", time.time() - start)

start = time.time()
ME.model(tr_x, tr_y, va_x, va_y)
print("--3--->", time.time() - start)
ray.shutdown()