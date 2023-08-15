# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
import sys
sys.path.append("..")

import pandas as pd
from quarkml.feature_engineering import FeatureEngineering
from quarkml.model_engineering import ModelEngineering
FE = FeatureEngineering()
ME = ModelEngineering()

import ray
# ray start --head --port=1063 --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265
runtime_envs = {"working_dir": ".."}
context = ray.init(runtime_env = runtime_envs)
print(context.dashboard_url)

# step1
ds = pd.read_csv("../experiment/credit/credit.csv")
ds, cat, con = FE.data_processing(ds, 'class', is_fillna=True, verbosity=False)
print("-1->", ds)
# # step1.1
# testds = pd.read_csv("../experiment/credit/credit-g.arff")
# ds = FE.data_processing(testds, 'class', task='tranform', verbosity=False)
# print("-2->", ds)
# step2
X = ds.drop('class', axis=1)
y = ds[['class']]


# selected_feature = FE.feature_generation(X, cat, con)

# step3.1
# selected_feature, ds = FE.feature_selector(X, y, selected_feature, cat, con, method='booster', distributed_and_multiprocess=2)
# print("-3.1->", selected_feature)

# step3.2
# selected_feature, ds = FE.feature_selector(X, y, None, cat, con, method='tmodel', distributed_and_multiprocess=1)
# print("-3.2->", selected_feature)

# step3.3
# selected_feature, ds = FE.feature_selector(X, y, selected_feature, cat, con, method='fwiz')
# print("-3.3->", selected_feature)

# step3.4
# selected_feature, ds = FE.feature_selector(X, y, None, cat, con, part_column='age', method='iv', distributed_and_multiprocess=-1)
# print("-3.4->", selected_feature)

# step3.5
# selected_feature, ds = FE.feature_selector(X, y, None, cat, con, part_column='age', method='psi', distributed_and_multiprocess=2)
# print("-3.5->", selected_feature)

# step4.1
# best_params = ME.hparams(X, y, method='hyperopt')
# print("-4.1-->", best_params)

# step4.2
# best_params = ME.hparams(X, y, method='optuna')
# print("-4.2-->", best_params)

# step5.1
# ME.model_cv(X, y, distributed_and_multiprocess=-1, params=best_params)

# step5.2
# ME.model(X, y, params=best_params)


ray.shutdown()