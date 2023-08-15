# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
import sys
sys.path.append("..")
# ray start --head --port=1063 --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265
import ray

from quarkml.feature_engineering import FeatureEngineering
from quarkml.model_engineering import ModelEngineering
FE = FeatureEngineering()
ME = ModelEngineering()
runtime_envs = {"working_dir": ".."}
context = ray.init(runtime_env = runtime_envs)
print(context.dashboard_url)

# 读取数据
# step1
# ds = pd.read_csv("../experiment/credit/credit.csv")
# ds, cat, con = FE.data_processing(ds, 'class', is_fillna=True, verbosity=False)
# print("-1->", ds)
# tr_ds = ds[:800]
# va_ds = ds[800:]

# tr_x = tr_ds.drop('class', axis=1)
# tr_y = tr_ds[['class']]
# va_x = va_ds.drop('class', axis=1)
# va_y = va_ds[['class']]

# tr_ds = FE.dist_data_processing(ds=tr_ds)
# va_ds = FE.dist_data_processing(ds=va_ds)
# print("-1->", tr_ds.schema().types)
# print("-1.1->", tr_ds.count())
# print("-2->", tr_ds.take(4))

# ME.dist_model(tr_ds, 'class', va_ds)


# ME.model(tr_x,tr_y, va_x, va_y)

ds, categorical_features, numerical_features = FE.dist_data_processing("experiment/credit/credit.csv", 'class')

ME.dist_model(ds, 'class', categorical_features=categorical_features)

ray.shutdown()