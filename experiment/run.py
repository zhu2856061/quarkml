# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
import sys

sys.path.append("..")

import ray
import pandas as pd
from quarkml.feature_engineering import FeatureEngineering
from quarkml.model_engineering import ModelEngineering

FE = FeatureEngineering()
ME = ModelEngineering()

######### 启动ray 构建一个单机环境 , 单机自己测试就启动这个后再运行程序，若分布式多机，则分布式环境已经有了，无需启动
# ray start --head --port=1063 --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265
#########

######### 设置ray 的运行时相关环境 #########
# address 集群地址，若是本地机器在分布式环境中，可以用 auto， 若是不在则需指定head节点 address="ray://123.45.67.89:10001"
# runtime_env 运行环境， working_dir 指定工作目录
context = ray.init(
    address="auto",
    runtime_env={"working_dir": ".."},
    ignore_reinit_error=True,
    include_dashboard=True,
    dashboard_host='127.0.0.1',
    dashboard_port='8265',
)
print(context.dashboard_url)
#########

######### 数据处理 #########
# step1 对数据进行处理
ds = pd.read_csv("../experiment/credit/credit.csv")
ds, cat, con = FE.data_processing(ds, 'class', is_fillna=True, verbosity=False)
# step1.1 基于新数据采用同样的处理方法
ds = pd.read_csv("../experiment/credit/credit.csv")
ds, cat, con = FE.data_processing(ds, 'class', is_fillna=True, verbosity=False)
#########

######### 特征衍生 #########
# step2
X = ds.drop('class', axis=1)
y = ds[['class']]
selected_feature = FE.feature_generation(X, cat, con)
#########

######### 特征选择 #########
# step3.1 booster
selected_feature, ds = FE.feature_selector(
    X,
    y,
    selected_feature,
    cat,
    con,
    method='booster',
    distributed_and_multiprocess=2,
)
print("-3.1->", selected_feature)

# step3.2 fwiz
selected_feature, ds = FE.feature_selector(
    X,
    y,
    selected_feature,
    cat,
    con,
    method='fwiz',
)

# step3.3 iv
selected_feature, ds = FE.feature_selector(
    X,
    y,
    None,
    cat,
    con,
    part_column='age',
    method='iv',
    distributed_and_multiprocess=2,
)

# step3.4 psi
selected_feature, ds = FE.feature_selector(
    X,
    y,
    None,
    cat,
    con,
    part_column='age',
    method='psi',
    distributed_and_multiprocess=2,
)

# step3.5 tmodel
selected_feature, ds = FE.feature_selector(
    X,
    y,
    selected_feature,
    cat,
    con,
    method='tmodel',
    distributed_and_multiprocess=2,
)
#########

######### 自动超参 #########
# step4.1 hyperopt
best_params_hyperopt = ME.hparams(ds, y, method='hyperopt')
print("-4.1-->", best_params_hyperopt)

# step4.2 optuna
best_params_optuna = ME.hparams(ds, y, method='optuna')
print("-4.2-->", best_params_optuna)
#########

######### 训练 #########
# step5.1 交叉验证
ME.model_cv(X, y, params=best_params_hyperopt, distributed_and_multiprocess=2)

# step5.2 训练
ME.model(X, y, params=best_params_hyperopt)
#########

# ******************************************************************************
######### 分布式 #########
# ******************************************************************************

######### 数据读取 #########
ds, categorical_features, numerical_features = FE.dist_data_processing(
    "experiment/credit/credit.csv", 'class')

######### 训练 #########
ME.dist_model(ds, 'class', categorical_features=categorical_features)

ray.shutdown()