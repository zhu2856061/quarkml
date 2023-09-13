# -*- coding: utf-8 -*-
# @Time   : 2023/5/29 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
from __future__ import absolute_import, division, print_function

import treelite
from treelite_runtime import Predictor, DMatrix

class Predict(object):

    def __init__(self) -> None:
        pass

    def compile_model(self, model_path, model_format='lightgbm', jobs=8):
        model = treelite.Model.load(model_path, model_format=model_format)
        # toolchain: 当前环境是Ubuntu，使用gcc进行编译
        # libpath: 导出的目标位置
        # params: 模型文件比较大，编译花费的时间比较多，开启并行编译
        model.export_lib(
            toolchain='gcc',
            libpath=model_path + '_lite.so',
            params={'parallel_comp': jobs},
        )

    def load_model_so(self, model_path):
        self.model = Predictor(model_path)

    def predict_x(self, x):
        return self.model.predict(DMatrix(x))