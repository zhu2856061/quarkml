# -*- coding: utf-8 -*-
# @Time   : 2023/7/27 09:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
# type: ignore
import os
import gc
import sys
import numpy as np
import pandas as pd
from loguru import logger
import ray
from ray.util.multiprocessing import Pool
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score
import scipy.special
import random
import traceback
from typing import List
from quarkml.model.tree_model import lgb_train
from quarkml.utils import Node, tree_to_formula, formula_to_tree, to_csv, get_categorical_numerical_features, transform, error_callback
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


class BoosterSelector(object):

    def __init__(self) -> None:
        self.tmp_save_path = '/tmp/booster_tmp_data.feather'

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        candidate_features: List[str],
        categorical_features: List[str] = None,
        params=None,
        select_method='predictive',
        min_candidate_features=200,
        blocks=2,
        ratio=0.5,
        seed=2023,
        report_dir="encode",
        distributed_and_multiprocess=-1,
        job=-1,
    ):
        self.select_method = select_method
        self.seed = seed
        self.y = y
        assert candidate_features is not None
        random.seed(self.seed)
        if job < 0:
            job = os.cpu_count()

        candidate_features = [formula_to_tree(_) for _ in candidate_features]

        if categorical_features is not None:
            categorical_features, _ = get_categorical_numerical_features(X)
        for cate_fea in categorical_features:
            try:
                X[cate_fea] = X[cate_fea].astype('category')
            except:
                continue
        # 使用to_feather保存数据，提升读取效率
        X.index.name = 'openfe_index'
        X.reset_index().to_feather(self.tmp_save_path)

        self.metric = 'auc'
        # self.metric = 'binary_logloss'
        if params is not None:
            self.metric = params.get('metric', 'binary_logloss')

        # 获得原始的初始y_pred 分
        self.init_scores = self._get_init_score(X, y, params, seed=seed)


        # 拿到样本下标， 按block 划分
        _, _, train_y, test_y = train_test_split(X,
                                                 y,
                                                 test_size=0.2,
                                                 random_state=seed)

        train_index_samples = self._subsample(train_y.index, blocks)
        val_index_samples = self._subsample(test_y.index, blocks)

        # 单特征提升法
        idx = 0
        train_idx = train_index_samples[idx]
        val_idx = val_index_samples[idx]
        idx += 1

        results = self._calculate_and_evaluate(
            candidate_features,
            train_idx,
            val_idx,
            params,
            distributed_and_multiprocess,
            job,
        )
        candidate_features_scores = sorted(results,
                                           key=lambda x: x[1],
                                           reverse=True)
        # 剔除掉两两差异不大的特征
        candidate_features_scores = self._delete_same(
            candidate_features_scores)

        while idx < len(train_index_samples):
            n_reserved_features = max(
                int(len(candidate_features_scores) * ratio),
                min(len(candidate_features_scores), min_candidate_features))

            train_idx = train_index_samples[idx]
            val_idx = val_index_samples[idx]
            idx += 1

            if n_reserved_features <= min_candidate_features:
                train_idx = train_index_samples[-1]
                val_idx = val_index_samples[-1]
                idx = len(train_index_samples)
                logger.info(
                    "Meet early-stopping in successive feature-wise halving.")

            candidate_features_list = [
                item[0]
                for item in candidate_features_scores[:n_reserved_features]
            ]

            del candidate_features_scores[n_reserved_features:]
            gc.collect()

            # 计算分
            results = self._calculate_and_evaluate(
                candidate_features_list,
                train_idx,
                val_idx,
                params,
                distributed_and_multiprocess,
                job,
            )

            # 倒排
            candidate_features_scores = sorted(results,
                                               key=lambda x: x[1],
                                               reverse=True)

            # 剔除
            candidate_features_scores = self._delete_same(
                candidate_features_scores)

        # 取正收益的
        candidate_features_scores = [
            _ for _ in candidate_features_scores if _[1] > 0
        ]
        return_results = [
            item[0] for item in candidate_features_scores if item[1] > 0
        ]

        # 若都为负分数，则取top 100
        if not return_results:
            return_results = [
                item[0] for item in candidate_features_scores[:100]
            ]

        os.remove(self.tmp_save_path)
        gc.collect()

        to_csv(candidate_features_scores, report_dir)
        selected_fea = [tree_to_formula(fea) for fea in return_results]
        if len(selected_fea) > 0:
            ds, _ = transform(X, selected_fea)
            return selected_fea, ds
        else:
            return selected_fea, X

    def _subsample(self, iterators, n_data_blocks):
        # 基于iterators 返回list block-> block大小是递增
        # iterators 若len为100 , n_data_blocks = 5，则
        # list -> [ [0:20], [0:40], [0:80], [0:100] ]
        iterators = list(iterators)
        length = int(len(iterators) / n_data_blocks)
        random.shuffle(iterators)
        results = [iterators[:length]]
        while n_data_blocks != 1:
            n_data_blocks = int(n_data_blocks / 2)
            length = int(length * 2)
            if n_data_blocks == 1:
                results.append(iterators)
            else:
                results.append(iterators[:length])

        return results

    def _calculate_and_evaluate(
        self,
        candidate_features: List[Node],
        train_idx,
        val_idx,
        params,
        distributed_and_multiprocess=-1,
        n_jobs=1,
    ):

        length = int(np.ceil(len(candidate_features) / n_jobs / 16))
        n = int(np.ceil(len(candidate_features) / length))
        random.shuffle(candidate_features)
        for f in candidate_features:
            f.delete()

        if distributed_and_multiprocess == 1:
            _calculate_and_evaluate_multiprocess_remote = ray.remote(
                _calculate_and_evaluate_multiprocess)
        elif distributed_and_multiprocess == 2:
            pool = Pool(n_jobs)

        train_y = self.y.iloc[train_idx]
        val_y = self.y.iloc[val_idx]
        train_init = self.init_scores.loc[train_idx]
        val_init = self.init_scores.loc[val_idx]
        
        futures_list = []
        for i in range(n):
            if i == (n - 1):
                if distributed_and_multiprocess == 1:
                    futures = _calculate_and_evaluate_multiprocess_remote.remote(
                        self.select_method,
                        self.metric,
                        candidate_features[i * length:],
                        train_idx,
                        val_idx,
                        params,
                        train_y,
                        val_y,
                        train_init,
                        val_init,
                        i,
                    )
                elif distributed_and_multiprocess == 2:
                    
                    futures = pool.apply_async(
                        _calculate_and_evaluate_multiprocess, (
                            self.select_method,
                            self.metric,
                            candidate_features[i * length:],
                            train_idx,
                            val_idx,
                            params,
                            self.tmp_save_path,
                            train_y,
                            val_y,
                            train_init,
                            val_init,
                            i,
                        ), error_callback=error_callback)
                else:
                    futures = _calculate_and_evaluate_multiprocess(
                        self.select_method,
                        self.metric,
                        candidate_features[i * length:],
                        train_idx,
                        val_idx,
                        params,
                        self.tmp_save_path,
                        train_y,
                        val_y,
                        train_init,
                        val_init,
                        i,
                    )
                futures_list.append(futures)
            # else:
            #     if distributed_and_multiprocess == 1:
            #         futures = _calculate_and_evaluate_multiprocess_remote.remote(
            #             self.select_method,
            #             self.metric,
            #             candidate_features[i * length:(i + 1) * length],
            #             train_idx,
            #             val_idx,
            #             params,
            #             train_y,
            #             val_y,
            #             train_init,
            #             val_init,
            #             i,
            #         )
            #     elif distributed_and_multiprocess == 2:
            #         futures = pool.apply_async(
            #             _calculate_and_evaluate_multiprocess, (
            #                 self.select_method,
            #                 self.metric,
            #                 candidate_features[i * length:(i + 1) * length],
            #                 train_idx,
            #                 val_idx,
            #                 params,
            #                 self.tmp_save_path,
            #                 train_y,
            #                 val_y,
            #                 train_init,
            #                 val_init,
            #                 i,
            #             ), error_callback=error_callback)
            #     else:
            #         futures = _calculate_and_evaluate_multiprocess(
            #             self.select_method,
            #             self.metric,
            #             candidate_features[i * length:(i + 1) * length],
            #             train_idx,
            #             val_idx,
            #             params,
            #             self.tmp_save_path,
            #             train_y,
            #             val_y,
            #             train_init,
            #             val_init,
            #             i,
            #         )
            #     futures_list.append(futures)

        if distributed_and_multiprocess == 2:
            pool.close()
            pool.join()

        result = []
        if distributed_and_multiprocess == 1:
            for _ in ray.get(futures_list):
                result.extend(_)
        elif distributed_and_multiprocess == 2:
            for _ in futures_list:
                result.extend(_.get())
        else:
            for _ in futures_list:
                result.extend(_)
        return result

    def _delete_same(self, candidate_features_scores, threshold=1e-20):
        start_n = len(candidate_features_scores)
        if candidate_features_scores:
            pre_score = candidate_features_scores[0][1]
        else:
            return candidate_features_scores
        i = 1
        while i < len(candidate_features_scores):
            now_score = candidate_features_scores[i][1]
            if abs(now_score - pre_score) < threshold:
                candidate_features_scores.pop(i)
            else:
                pre_score = now_score
                i += 1
        end_n = len(candidate_features_scores)
        logger.info(f"{start_n-end_n} same features have been deleted.")
        return candidate_features_scores

    def _get_init_score(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        params=None,
        feature_boosting=True,
        seed=2023,
    ):
        # 获得初始的分值，即原始特征，经过lgb训练得到的basic scores
        # 两种方法： 一种是采用提升法训练出来的初始分，一种是直接对所有的y 进行取均值
        if feature_boosting:
            # boosting 将利用lgb 获得初始分
            nuti_class = False
            if y[y.columns[0]].nunique() > 2:
                init_scores = np.zeros((len(X), y[y.columns[0]].nunique()))
                skf = StratifiedKFold(n_splits=5, random_state=seed)
                nuti_class = True
            else:
                init_scores = np.zeros(len(X))
                skf = KFold(n_splits=5)

            for _, (t_index, v_index) in enumerate(skf.split(
                    X, y)):  # 核心：必须进行交叉才能得到所有的数据预估score
                trn_x = X.iloc[t_index]
                trn_y = y.iloc[t_index]
                val_x = X.iloc[v_index]
                val_y = y.iloc[v_index]

                futures = lgb_train(
                    trn_x,
                    trn_y,
                    val_x,
                    val_y,
                    None,
                    params,
                )

                init_scores[v_index] = futures[0].predict_proba(
                    val_x, raw_score=True
                ) if nuti_class else futures[0].predict(val_x)

            init_scores = pd.DataFrame(init_scores, index=y.index)

        else:
            # 利用真实的label 均值作为初始分
            if params['objective'] == 'regression':
                init_scores = np.array([np.mean(y.values.ravel())] * len(y))
            else:
                prob = y[y.columns[0]].value_counts().sort_index().to_list()
                prob = prob / np.sum(prob)
                prob = [list(prob)]
                init_scores = np.array(prob * len(y))

            init_scores = pd.DataFrame(init_scores, index=y.index)

        return init_scores


def _calculate_and_evaluate_multiprocess(
    select_method,
    metric,
    candidate_features: List[Node],
    train_idx,
    val_idx,
    params,
    tmp_save_path,
    train_y,
    val_y,
    train_init,
    val_init,
    # y: pd.DataFrame,
    # init_scores: pd.DataFrame,
    i,
):
    try:
        results = []
        base_features = {'openfe_index'}
        for candidate_feature in candidate_features:
            base_features |= set(candidate_feature.get_fnode())
        
        data = pd.read_feather(
            tmp_save_path,
            columns=list(base_features)).set_index('openfe_index')
        data_temp = data.iloc[train_idx + val_idx]
        del data
        gc.collect()

        # train_y = y.iloc[train_idx]
        # val_y = y.iloc[val_idx]
        # train_init = init_scores.loc[train_idx]
        # val_init = init_scores.loc[val_idx]
        init_metric = _get_init_metric(metric, val_init, val_y)
        for candidate_feature in candidate_features[:1]:
            candidate_feature.calculate(data_temp, is_root=True)
            
            # score = _evaluate(select_method, metric, candidate_feature,
            #                   train_y, val_y, params, train_init, val_init,
            #                   init_metric)
            

            train_x = pd.DataFrame(candidate_feature.data.loc[train_y.index])
            val_x = pd.DataFrame(candidate_feature.data.loc[val_y.index])
            if select_method == 'predictive':
                """基于之前的basic 【train_init， val_init】预测分 基础上进行继续训练，
                    而继续训练只采样单特征进行，从而确保，在原来的基础上增益来自该特征
                """
                if params is None:
                    params = {'period': 2000, 'n_estimators': 200, 'stopping_rounds': 20}  # 为了不打印
                futures = lgb_train(
                    train_x,
                    train_y,
                    val_x,
                    val_y,
                    params=params,
                    trn_init_score=train_init,
                    val_init_score=val_init,
                )

                key = list(futures[0].best_score_['valid_1'].keys())[0]
                if metric in ['auc']:
                    score = futures[0].best_score_['valid_1'][key] - init_metric
                else:
                    score = init_metric - futures[0].best_score_['valid_1'][key]

            elif select_method == 'corr':
                score = np.corrcoef(
                    pd.concat([train_x, val_x], axis=0).fillna(0).values.ravel(),
                    pd.concat([train_y, val_y],
                                axis=0).fillna(0).values.ravel())[0, 1]
                score = abs(score)
            else:
                raise NotImplementedError("Cannot recognize select_method %s." %
                                            select_method)

            candidate_feature.delete()
            results.append([candidate_feature, score])
        logger.info(
            f'************************************ {i + 1} end ************************************'
        )
        return results
    except:
        print(traceback.format_exc())
        sys.exit()


def _evaluate(
    select_method,
    metric,
    candidate_feature: Node,
    train_y: pd.DataFrame,
    val_y: pd.DataFrame,
    params,
    train_init,
    val_init,
    init_metric,
):
    try:
        train_x = pd.DataFrame(candidate_feature.data.loc[train_y.index])
        val_x = pd.DataFrame(candidate_feature.data.loc[val_y.index])
        if select_method == 'predictive':
            """基于之前的basic 【train_init， val_init】预测分 基础上进行继续训练，
                而继续训练只采样单特征进行，从而确保，在原来的基础上增益来自该特征
            """
            if params is None:
                params = {'period': 2000}  # 为了不打印
            futures = lgb_train(
                train_x,
                train_y,
                val_x,
                val_y,
                params=params,
                trn_init_score=train_init,
                val_init_score=val_init,
            )

            key = list(futures[0].best_score_['valid_1'].keys())[0]
            if metric in ['auc']:
                score = futures[0].best_score_['valid_1'][key] - init_metric
            else:
                score = init_metric - futures[0].best_score_['valid_1'][key]

        elif select_method == 'corr':
            score = np.corrcoef(
                pd.concat([train_x, val_x], axis=0).fillna(0).values.ravel(),
                pd.concat([train_y, val_y],
                            axis=0).fillna(0).values.ravel())[0, 1]
            score = abs(score)
        else:
            raise NotImplementedError("Cannot recognize select_method %s." %
                                        select_method)
        return score
    except:
        print(traceback.format_exc())
        sys.exit()


def _get_init_metric(metric, pred, label):
    if metric == 'binary_logloss':
        init_metric = log_loss(label, scipy.special.expit(pred), labels=[0, 1])
    elif metric == 'multi_logloss':
        init_metric = log_loss(label,
                               scipy.special.softmax(pred, axis=1),
                               labels=list(range(pred.shape[1])))
    elif metric == 'rmse':
        init_metric = mean_squared_error(label, pred, squared=False)
    elif metric == 'auc':
        init_metric = roc_auc_score(label, scipy.special.expit(pred))
    else:
        raise NotImplementedError(
            f"Metric {metric} is not supported. "
            f"Please select metric from ['binary_logloss', 'multi_logloss'"
            f"'rmse', 'auc'].")
    return init_metric


