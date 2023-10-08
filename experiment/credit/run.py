# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function
import sys

sys.path.append("../..")

import ray
import pandas as pd
from quarkml.feature_engineering import FeatureEngineering
from quarkml.model_engineering import ModelEngineering
from quarkml.distributed_engineering import DistributedEngineering

######### 启动ray 构建一个单机环境 , 单机自己测试就启动这个后再运行程序，若分布式多机，则分布式环境已经有了，无需启动
# ray start --head --port=1063 --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265
#########

######### 设置ray 的运行时相关环境 #########
# address 集群地址，若是本地机器在分布式环境中，可以用 auto， 若是不在则需指定head节点 address="ray://123.45.67.89:10001"
# runtime_env 运行环境， working_dir 指定工作目录
context = ray.init(
    address="auto",
    runtime_env={"working_dir": "../.."},
    ignore_reinit_error=True,
    include_dashboard=True,
    dashboard_host='127.0.0.1',
    dashboard_port='8265',
)
print(context.dashboard_url)
#########
FE = FeatureEngineering()
ME = ModelEngineering()
DE = DistributedEngineering()
######### 数据处理 #########
# file_path = FE.data_processing_fit("credit.csv", 'class')
# ds = pd.read_csv("credit.csv")
# ds, cat, con = FE.data_processing_fit(ds, 'class')
# print(ds)
# # step1.1 基于新数据采用同样的处理方法
# ds = pd.read_csv("credit.csv")
# ds, cat, con = FE.data_processing_transform(ds, 'class')
# print(ds)
#########

######### 特征衍生 #########
# file_path = FE.feature_generation("credit.csv", 'class', is_filter=True)
# ds = FE.feature_generation(ds, 'class', cat, is_filter=True)
#########

# ######### 特征选择 #########
# ds = FE.feature_selector(ds, 'class', part_column='age', cate_features=cat,  method='tmodel')
# #########

# ######### 自动超参 #########
# best_params_hyperopt = ME.hparams(ds, 'class', cat_features=cat)
# #########

# ######### 交叉验证 #########
# ME.model_cv(ds, 'class')
# #########

# ######### 训练 #########
# cls = ME.model(ds, 'class')
# #########

# ######### 可解释性 #########
# x = ds.drop('class', axis=1)
# ME.interpretable('regression', cls[0], X, single_index=1)
# #########

# # ******************************************************************************
# ######### 分布式 #########
# # ******************************************************************************

# ######### 数据读取 #########
# ds, categorical_features, numerical_features = FE.dist_data_processing(
#     "experiment/credit/credit.csv", 'class')

# ######### 训练 #########
FE.data_processing_fit("credit.csv", 'class')
DE.dist_model("experiment/credit/credit.csv_data_processing.csv", 'class')
ray.shutdown()