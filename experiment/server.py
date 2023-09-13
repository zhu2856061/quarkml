# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import json
import pandas as pd
import ray
from loguru import logger
from flask import Flask, request
from quarkml.model_engineering import ModelEngineering
from quarkml.feature_engineering import FeatureEngineering

app = Flask(__name__)
FE = FeatureEngineering()
ME = ModelEngineering()

context = ray.init(
    address="auto",
    runtime_env={"working_dir": "."},
    ignore_reinit_error=True,
    include_dashboard=True,
    dashboard_host='127.0.0.1',
    dashboard_port='8265',
)
print(context.dashboard_url)

# {data_file: "", label_name: ""}
@app.route("/data_processing", methods=["POST"])
def run_data_processing():
    logger.info(f">>>>> request data: {request.get_data()} <<<<<")
    if request.get_data() is None:
        return_dict = {"errcode": 1, "errdesc": "req data is None"}
        return json.dumps(return_dict)
    try:
        data = json.loads(request.get_data())
        ds = pd.read_csv(data['data_file'])
        ds, _, _ = FE.data_processing(
            ds=ds,
            label_name=data['label_name'],
            is_fillna=True,
            verbosity=True,
        )
        ds.to_csv(data['data_file'] + "_dp.csv", index=False)
        return json.dumps({"errcode": 0, "errdesc": "文件已经处理完成"})
    except Exception as e:
        logger.info(f"异常类型为：{e}")
        return_dict = {"errcode": 1, "errdesc": f"req data is err: {e}"}
        return json.dumps(return_dict)

# {'data_file': "", 'label_name': "", 'is_data_processing': 1}
@app.route("/feature_generation", methods=["POST"])
def run_feature_generation():
    logger.info(f">>>>> request data: {request.get_data()} <<<<<")
    if request.get_data() is None:
        return_dict = {"errcode": 1, "errdesc": "req data is None"}
        return json.dumps(return_dict)
    try:
        data = json.loads(request.get_data())
        ds = pd.read_csv(data['data_file'])
        cat, con = None, None
        if data.get('is_data_processing', 0) == 1:
            ds, cat, con = FE.data_processing(
                ds=ds,
                label_name=data['label_name'],
                is_fillna=True,
                verbosity=True,
            )
        y = ds[[data['label_name']]]
        X = ds.drop(data['label_name'], axis=1)
        # 生成
        selected_feature = FE.feature_generation(X, cat, con)

        # booster 筛选
        selected_feature, ds = FE.feature_selector(
            X=X,
            y=y,
            candidate_features=selected_feature,
            categorical_features=cat,
            numerical_features=con,
            method='booster',
            distributed_and_multiprocess=2,
        )
        logger.info(f">>>>>{selected_feature}<<<<<")
        ds.to_csv(data['data_file'] + "_fg.csv", index=False)
        return json.dumps({"errcode": 0, "errdesc": "特征衍生完成"})
    except Exception as e:
        logger.info(f"异常类型为：{e}")
        return_dict = {"errcode": 1, "errdesc": f"req data is err: {e}"}
        return json.dumps(return_dict)

# {data_file: "", label_name: "", method: "A", 'is_data_processing': 1, "part_column": "col_name"}
@app.route("/feature_selector", methods=["POST"])
def run_feature_selector():
    logger.info(f">>>>> request data: {request.get_data()} <<<<<")
    if request.get_data() is None:
        return_dict = {"errcode": 1, "errdesc": "req data is None"}
        return json.dumps(return_dict)
    try:
        data = json.loads(request.get_data())
        ds = pd.read_csv(data['data_file'])
        cat, con = None, None
        if data.get('is_data_processing', 0) == 1:
            ds, cat, con = FE.data_processing(
                ds=ds,
                label_name=data['label_name'],
                is_fillna=True,
                verbosity=True,
            )
        y = ds[[data['label_name']]]
        X = ds.drop(data['label_name'], axis=1)

        method = data['method']
        if method == "fwiz":
            # step3.2 fwiz
            selected_feature, ds = FE.feature_selector(
                X=X,
                y=y,
                candidate_features=cat,
                numerical_features=con,
                method='fwiz',
            )
            logger.info(f"-3.2->{selected_feature}")
        if method == "iv":
            # step3.3 iv
            selected_feature, ds = FE.feature_selector(
                X=X,
                y=y,
                candidate_features=cat,
                numerical_features=con,
                method='iv',
                distributed_and_multiprocess=2,
            )
            logger.info(f"-3.3->{selected_feature}")
        if method == "psi":
            # step3.4 psi
            part_column = data['part_column']
            selected_feature, ds = FE.feature_selector(
                X=X,
                y=y,
                candidate_features=cat,
                numerical_features=con,
                part_column=part_column,
                method='psi',
                distributed_and_multiprocess=2,
            )
            logger.info(f"-3.4->{selected_feature}")
        if method == "tmodel":
            # step3.5 tmodel
            selected_feature, ds = FE.feature_selector(
                X=X,
                y=y,
                candidate_features=cat,
                numerical_features=con,
                method='tmodel',
                distributed_and_multiprocess=2,
            )
            logger.info(f"-3.5->{selected_feature}")

        return json.dumps({"errcode": 0, "errdesc": ','.json(selected_feature)})
    except Exception as e:
        logger.info(f"异常类型为：{e}")
        return_dict = {"errcode": 1, "errdesc": f"req data is err: {e}"}
        return json.dumps(return_dict)

# {data_file: "", label_name: ""}
@app.route("/model_cv", methods=["POST"])
def run_model_cv():
    logger.info(f">>>>> request data: {request.get_data()} <<<<<")
    if request.get_data() is None:
        return_dict = {"errcode": 1, "errdesc": "req data is None"}
        return json.dumps(return_dict)
    try:
        data = json.loads(request.get_data())
        ds = pd.read_csv(data['data_file'])
        ds, cat, con = FE.data_processing(
            ds,
            data['label_name'],
            is_fillna=True,
            verbosity=True,
        )
        y = ds[[data['label_name']]]
        X = ds.drop(data['label_name'], axis=1)
        ME.model_cv(X, y, categorical_features=cat)
        return json.dumps({"errcode": 0, "errdesc": "ok"})
    except Exception as e:
        logger.info(f"异常类型为：{e}")
        return_dict = {"errcode": 1, "errdesc": f"req data is err: {e}"}
        return json.dumps(return_dict)

# {data_file: "", label_name: ""}
@app.route("/model", methods=["POST"])
def run_model():
    logger.info(f">>>>> request data: {request.get_data()} <<<<<")
    if request.get_data() is None:
        return_dict = {"errcode": 1, "errdesc": "req data is None"}
        return json.dumps(return_dict)
    try:
        data = json.loads(request.get_data())
        ds = pd.read_csv(data['data_file'])
        ds, cat, con = FE.data_processing(
            ds=ds,
            label_name=data['label_name'],
            is_fillna=True,
            verbosity=True,
        )
        y = ds[[data['label_name']]]
        X = ds.drop(data['label_name'], axis=1)
        ME.model(X, y, categorical_features=cat)
        return json.dumps({"errcode": 0, "errdesc": "ok"})
    except Exception as e:
        logger.info(f"异常类型为：{e}")
        return_dict = {"errcode": 1, "errdesc": f"req data is err: {e}"}
        return json.dumps(return_dict)

# {data_file: "", label_name: ""}
@app.route("/hparams", methods=["POST"])
def run_hparams():
    logger.info(f">>>>> request data: {request.get_data()} <<<<<")
    if request.get_data() is None:
        return_dict = {"errcode": 1, "errdesc": "req data is None"}
        return json.dumps(return_dict)
    try:
        data = json.loads(request.get_data())
        ds = pd.read_csv(data['data_file'])
        ds, cat, con = FE.data_processing(
            ds,
            data['label_name'],
            is_fillna=True,
            verbosity=True,
        )
        X = ds.drop(data['label_name'], axis=1)
        y = ds[[data['label_name']]]
        best_params_hyperopt = ME.hparams(X, y, method='hyperopt')
        return json.dumps({"errcode": 0, "errdesc": str(best_params_hyperopt)})
    except Exception as e:
        logger.info(f"异常类型为：{e}")
        return_dict = {"errcode": 1, "errdesc": f"req data is err: {e}"}
        return json.dumps(return_dict)

# {data_file: "", label_name: ""}
@app.route("/automl", methods=["POST"])
def get_automl():
    logger.info(f">>>>> request data: {request.get_data()} <<<<<")
    if request.get_data() is None:
        return_dict = {"errcode": 1, "errdesc": "req data is None"}
        return json.dumps(return_dict)
    try:
        data = json.loads(request.get_data())
        ds = pd.read_csv(data['data_file'])
        ds, cat, con = FE.data_processing(
            ds,
            data['label_name'],
            is_fillna=True,
            verbosity=True,
        )
        y = ds[[data['label_name']]]
        X = ds.drop(data['label_name'], axis=1)
        selected_feature = FE.feature_generation(X, cat, con)
        selected_feature, ds = FE.feature_selector(
            X=X,
            y=y,
            candidate_features=selected_feature,
            categorical_features=cat,
            numerical_features=con,
            method='booster',
            distributed_and_multiprocess=2,
        )
        selected_feature, ds = FE.feature_selector(
            X=X,
            y=y,
            candidate_features=selected_feature,
            categorical_features=cat,
            numerical_features=con,
            method='fwiz',
        )

        # step3.3 iv
        selected_feature, ds = FE.feature_selector(
            X=X,
            y=y,
            candidate_features=selected_feature,
            categorical_features=cat,
            numerical_features=con,
            method='iv',
            distributed_and_multiprocess=2,
        )

        # step3.5 tmodel
        selected_feature, ds = FE.feature_selector(
            X=X,
            y=y,
            candidate_features=selected_feature,
            categorical_features=cat,
            numerical_features=con,
            method='tmodel',
            distributed_and_multiprocess=2,
        )
        best_params_hyperopt = ME.hparams(ds, y, method='hyperopt')
        print("-4.1-->", best_params_hyperopt)
        ME.model_cv(X, y, params=best_params_hyperopt, distributed_and_multiprocess=2)

        # step5.2 训练
        ME.model(X, y, params=best_params_hyperopt)
        return json.dumps({"errcode": 0, "errdesc": str(best_params_hyperopt)})
        
    except Exception as e:
        logger.info(f"异常类型为：{e}")
        return_dict = {"errcode": 1, "errdesc": f"req data is err: {e}"}
        return json.dumps(return_dict)


if __name__ == "__main__":
    app.run(debug=True, port=12011, host="127.0.0.1")
