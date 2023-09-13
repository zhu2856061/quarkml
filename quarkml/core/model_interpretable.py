# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import pandas as pd
from loguru import logger
import shap
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


class ModelInterpretable(object):

    def __init__(self) -> None:
        shap.initjs()

    def init_set_model_x(self, model, X: pd.DataFrame):
        self.model = model
        self.X = X
        self.explainer = shap.Explainer(self.model)
        self.shap_values = self.explainer(self.X)
        # self.explainer = shap.TreeExplainer(model)
        # self.shap_values = self.explainer.shap_values(X)

    def single_prediction(self, index=0, task='regression'):

        if task == 'regression':
            # shap.force_plot(
            #     self.explainer.expected_value,
            #     self.shap_values[index, :],
            #     self.X.iloc[index, :],
            #     matplotlib=True,
            # )
            shap.plots.force(self.shap_values[index], matplotlib=True)
        else:
            for i, _ in enumerate(self.shap_values):
                logger.info(f"============= {i} =============")
                # shap.force_plot(
                #     self.explainer.expected_value[i],
                #     self.shap_values[i][index, :],
                #     self.X.iloc[index, :],
                #     matplotlib=True,
                # )
                shap.plots.force(self.shap_values[i][index], matplotlib=True)

    def many_prediction(self, num=100, task='regression'):
        if task == 'regression':
            # shap.force_plot(
            #     self.explainer.expected_value,
            #     self.shap_values[:num, :],
            #     self.X.iloc[:num, :],
            #     matplotlib=True,
            # )
            shap.plots.force(self.shap_values)
        else:
            for i, _ in enumerate(self.shap_values):
                logger.info(f"============= {i} =============")
                # shap.force_plot(
                #     self.explainer.expected_value[i],
                #     self.shap_values[i][:num, :],
                #     self.X.iloc[:num, :],
                #     matplotlib=True,
                # )
                shap.plots.force(self.shap_values)

    def single_waterfall(self, index=0, task='regression'):
        # explainer = shap.Explainer(self.model)
        # shap_values = explainer(self.X)
        if task == 'regression':
            shap.plots.waterfall(self.shap_values[index])
        else:
            for i, _ in enumerate(self.shap_values):
                logger.info(f"============= {i} =============")
                shap.plots.waterfall(self.shap_values[i][index])

    def many_waterfall(self, task='regression'):
        # explainer = shap.Explainer(self.model)
        # shap_values = explainer(self.X)
        if task == 'regression':
            shap.plots.beeswarm(self.shap_values)
        else:
            for i, _ in enumerate(self.shap_values):
                logger.info(f"============= {i} =============")
                shap.plots.beeswarm(self.shap_values[i])

    def sumary_prediction(self):
        shap.plots.bar(self.shap_values)
        # shap.summary_plot(self.shap_values, self.X, matplotlib=True)

    def feature_dependence(self, task):
        for name in self.X.columns:
            if task == 'regression':
                shap.dependence_plot(
                    name,
                    self.shap_values,
                    self.X,
                    display_features=self.X,
                    matplotlib=True,
                )
            else:
                for i, _ in enumerate(self.explainer.expected_value):
                    logger.info(f"============= {i} =============")
                    shap.dependence_plot(
                        name,
                        self.shap_values[i],
                        self.X,
                        display_features=self.X,
                        matplotlib=True,
                    )
