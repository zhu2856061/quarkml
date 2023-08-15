## quarkml

### 介绍
适用于风控算法领域，集 特征处理，特征工程，模型训练， 树模型分布式训练等功能于一体的自动化工具
主要功能：
    1. 特征工程
       a. 特征处理，分类特征转换数值index，空值填充，异常值去除, 特征分析报告打印
       b. 特征衍生，基于内置的算子进行自动衍生
       c. 特征选择，可对自身特征进行筛选，也可只对衍生特征进行筛选
    2. 模型工程
       a. 模型交叉验证，提供lgb多进程训练，分布式训练，常规训练，用于验证模型性能
       b. 模型训练，提供树模型多进程训练，分布式训练，常规训练
       c. 模型解释性，自动化产生shap 模型解释图

### 特征工程 FeatureEngineering 功能介绍

#### 数据预处理 data_processing()

```
ds: pd.DataFrame, 原始数据 （必选项）
label_name: str,  原始数据的label （必选项）
cat_feature: list = [], 指定类别特征， （非选项）， 若为空，则会利用原始数据的非number类型 设置 为cat
num_feature: list = [], 指定连续特征， （非选项）， 若为空，则会利用原始数据的number类型 设置 为num
ordinal_number=100, 指定数值特征若唯一值总数小于ordinal_number，则会划分成类别特征， （非选项）， 若为空，默认 100
report_dir="./encode", 对离散特征的值进行encode 编码，转换成数字index, 转换对应的映射关系为存储在report_dir路径下data_processing.pkl
is_fillna=True, 是否填充nan值，连续值填充众数，离散值填充均值
drop_outliers=True, 是否删除异常点
verbosity=True, 是否打印数据质量报告，html打开查看
task='fit', fit 为将数据处理，将生成新的数据样式，transform 为来一份新数据能基于同样的方法转换数据
```

使用样例(所有示例均可在experiment中运行测试)
``` python
FE = FeatureEngineering()
ME = ModelEngineering()
ds = pd.read_csv("../experiment/credit/credit.csv")
ds, cat, con = FE.data_processing(ds, 'class', is_fillna=True, verbosity=False)
```

#### 特征衍生 feature_generation

```
X: pd.DataFrame, 原始数据 （必选项）
categorical_features: list = None, 指定类别特征， （非选项）， 若为空，则会利用原始数据的非number类型 设置 为cat
numerical_features: list = None,   指定连续特征， （非选项）， 若为空，则会利用原始数据的number类型 设置 为num
report_dir: str = 'encode', 衍生产生特征的逻辑将存储在report_dir路径下 generation, 若为空，默认 encode
method: str = "basic", 衍生方法：目前就只有basic 方法 , 若为空，默认 basic
```

使用样例

``` python
X = ds.drop('class', axis=1)
y = ds[['class']]
candidate_features = FE.feature_generation(X, cat, con)
```

#### 特征选择 feature_selector
具备5中特征选择方法【booster，fwiz，iv，psi，tmodel】，其中除fwiz 方法外，其他方法均具备三种运行模式【常规，多进程，分布式】
小数据量下运行耗时H比较 H(多进程) < H(常规) < H(分布式)
大数据量下单机无法运行时 - 推荐用分布式

booster 提升法，该方法只能针对衍生特征进行，逻辑为，原始特征构建出数据的初始分，然后单衍生特征训练比较与初始化的提升量，若两两特征的提升量相似，则去掉其中一个，最后选择提升为正向的特征

fwiz 基于SULOV（搜索不相关的变量列表），SULOV 注定只能针对连续值，
SULOV算法基于本文中解释的最小冗余最大相关性（MRMR）算法，该算法是最佳特征选择方法之一

iv 风控领域常使用的IV值

psi 风控领域常使用的PSI值

tmodel 基于树模型训练后的特征重要性进行选择，其中有4种特征重要性【importance， permutation， shap， all】
    - importance 为树的划分节点gain值总和
    - permutation 评估特征的随机和非随机的预估值差值比较（模型无关性）
    - shap 类似permutation， 只是更加严格，【具体可学习Shap 值】
    - all 是综合上述三种方法选出的特征交集

```
X: pd.DataFrame,  原始数据X （必选项）
y: pd.DataFrame,  原始数据label （必选项）
candidate_features: list[str] = None,  候选特征，主要来自feature_generation方法衍生出来的特征，不为None的话，将在这个candidate_features内进行筛选，为None 的话，将在X.columns 的列表内筛选
categorical_features: list[str] = None,  指定类别特征， （非选项）， 若为空，则会利用原始数据的非number类型 设置 为cat
numerical_features: list[str] = None,    指定连续特征， （非选项）， 若为空，则会利用原始数据的number类型 设置 为num
params: dict = None,  lgb模型的参数，主要是用于booster 筛选方法和 tmodel 筛选方法，非必选项，若没有，会采用系统默认的
tmodel_method='mean',  tmodel_method方法会采用5折交叉得到重要性，因为会有5次重要性，选择mean即使对这5次进行均值后比大小，选择max即使选择最大的
init_score: pd.DataFrame = None,  初始分，用另一个模型分预测分作为模型的初始预估分，若为None，则会在booster 方法中自动训练一个初始分，只用于method为booster
importance_metric: str = 'importance',  tmodel_method方法得到的重要性分为4种方式的 ： importance， permutation， shap， all
select_method='predictive',  主要是用于booster 筛选方法中评估两个特征相似，是采用估计的方式predictive 还是corr 相关性的方式
min_candidate_features=200,  主要是用于booster 中最小候选数量的话，可以直接全量数据进行淘汰了，数据量越大，该值应该设置越小一般在200 - 2000
blocks=2,  主要是用于booster 中对数据进行增量式划分成多个数据块
ratio=0.5, 主要是用于booster 中candidate_features被淘汰ratio=0.5 一半了就，可以直接全量数据进行淘汰了，数据量越大，该值应该设置越小一般在0.2 ~ 0.5
folds=3, 交叉验证的数据划分份数，用于tmodel
seed=2023, 随机种子
part_column: str = None,  划分列，主要是用于PSI筛选特征的方法内，表明用这个一列进行划分数据集，然后比较每个数据两两之间的差异，当method为psi 时是必选项
bins=10,: IV计算中的分桶数

report_dir="encode", 特征筛选过程中的一些中间结果存在路径
method: str = "booster", 特征筛选的方法：fwiz ， iv ， psi ， tmodel， booster ， 注fwiz-基于SULOV（搜索不相关的变量列表）
distributed_and_multiprocess=-1， 三种运行模式【常规 -1，多进程 2，分布式 1】
job=-1,  主要是用于booster 中开启多个线程进行加快处理 -1 为cpu核数
```

使用样例
``` python
X = ds.drop('class', axis=1)
y = ds[['class']]
candidate_features = FE.feature_generation(X, cat, con)

# 对衍生的特征进行筛选
# step3.1
selected_feature, ds = FE.feature_selector(X, y, candidate_features, cat, con, part_column='age', method='booster')
print("-1.1->", selected_feature)

# step3.2
selected_feature, ds = FE.feature_selector(X, y, selected_feature, cat, con, part_column='age', method='tmodel')
print("-1.2->", selected_feature)

# step3.3
selected_feature, ds = FE.feature_selector(X, y, selected_feature, cat, con, part_column='age', method='fwiz')
print("-2->", selected_feature)

# step3.4
selected_feature, ds = FE.feature_selector(X, y, selected_feature, cat, con, part_column='age', method='iv')
print("-3->", selected_feature)

# step3.5
selected_feature, ds = FE.feature_selector(X, y, selected_feature, cat, con, part_column='age', method='psi')
print("-4->", selected_feature)

# -----------------------------------------------------------------
X = ds.drop('class', axis=1)
y = ds[['class']]
# 对自身特征进行筛选
# step3.2

selected_feature, ds = FE.feature_selector(X, y, None, cat, con, part_column='age', method='tmodel')
print("-1.2->", selected_feature)

# step3.3
selected_feature, ds = FE.feature_selector(X, y, None, cat, con, part_column='age', method='fwiz')
print("-2->", selected_feature)

# step5
selected_feature, ds = FE.feature_selector(X, y, None, cat, con, part_column='age', method='iv')
print("-3->", selected_feature)

# step6
selected_feature, ds = FE.feature_selector(X, y, None, cat, con, part_column='age', method='psi')
print("-4->", selected_feature)

```

### 模型工程 ModelEngineering

基于lgb的模型训练，交叉验证，基于shap的模型可解释性
基于ray的分布式，多进程训练

#### 模型交叉验证 model_cv
lgb 模型
```
X: pd.DataFrame,  原始数据X （必选项）
y: pd.DataFrame,  原始数据label （必选项）
train_index=None, 训练数据的下标， 不指定将直接对数据进行划分7：3 随机切分
test_index=None,  测试数据的下标
categorical_features: list = None, 指定类别特征， （非选项）， 若为空，则会利用原始数据的非number类型 设置 为cat
params=None, lgb模型的参数，非必选项，若没有，会采用系统默认的
folds=5,交叉验证的数据划分份数，用于tmodel
seed=2023,
distributed_and_multiprocess=-1， 三种运行模式【常规 -1，多进程 2，分布式 1】
job=-1,  主要是用于booster 中开启多个线程进行加快处理 -1 为cpu核数
```

使用样例
``` python
ME.model_cv(ds, y)
```

#### 模型训练 model
lgb 模型
```
X: pd.DataFrame,  原始数据X （必选项）
y: pd.DataFrame,  原始数据label （必选项）

train_index=None, 训练数据的下标， 不指定将直接对数据进行划分7：3 随机切分
test_index=None,  测试数据的下标
categorical_features: list = None, 指定类别特征， （非选项）， 若为空，则会利用原始数据的非number类型 设置 为cat
params=None, lgb模型的参数，非必选项，若没有，会采用系统默认的
seed=2023,
report_dir="encode" lgb模型训练保存的路径report_dir下的 loan_model.pkl
```

使用样例
``` python
ME.model(ds, y)
```

#### 模型解释性 model_interpretable
```
model, 训练完的lgb模型
X: pd.DataFrame, 不含 label 的原始数据
```

使用样例
``` python
ME.interpretable(tm, X)
```

----
### 【新增】分布式树模型训练，需对数据和模型进行改造

特征工程中的FeatureEngineering新增方法 dist_data_processing 分布式特征处理，并将数据转换成分布式数据样式，懒加载模式

#### 分布式特征处理 dist_data_processing
具备两种数据处理方式，第一种基于data_processing处理后的dataframe 数据进行直接转换
第二种基于ray.data 读取文本文件，直接分布式对象
```
files = None, 文件的路径，可以是多个文件list 也可是单个文件str， 注：必须是csv格式的数据
label_name: str = None,  数据的label列名
delimiter: str = ',',    数据切分符
cat_feature: list = [], 指定类别特征， （非选项）， 若为空，则会利用原始数据的非number类型 设置 为cat
num_feature: list = [], 指定连续特征， （非选项）， 若为空，则会利用原始数据的number类型 设置 为num
ordinal_number=100, 指定数值特征若唯一值总数小于ordinal_number，则会划分成类别特征， （非选项）， 若为空，默认 100
report_dir="./encode", 保留类别特征，连续特征 到路径下data_processing.pkl
ds: pd.DataFrame = None, 若设置该值也就是数据是来自data_processing处理后的dataframe, 则上述所有设置不起效， 采用第一种方式产生数据
```

使用样例
``` python
# 第一种方法，直接转换后dataframe 再转换成ray.data
ds = pd.read_csv("../experiment/credit/credit.csv")
ds, cat, con = FE.data_processing(ds, 'class', is_fillna=True, verbosity=False)
ds = FE.dist_data_processing(ds=ds)

# 第一种方法，直接转换成ray.data
ds, categorical_features, numerical_features = FE.dist_data_processing("experiment/credit/credit.csv", 'class')

```

#### 分布式训练 dist_model
模型工程中的ModelEngineering新增方法 dist_model 分布式训练
```
trn_ds, 训练数据 ray.data
label_name, 数据的label名
val_ds=None, 验证数据 ray.data
categorical_features = None, 类别特征列表
params=None, 模型参数，不设置会有模型的
seed=2023, 
num_workers=2, ray 工作空间，会对数据进行切分，数据分发
trainer_resources={"CPU": 4},  训练资源，会设置所有机器中使用多少个核用于纯训练计算
resources_per_worker={"CPU": 2}, 数据读取资源，会设置所有的工作空间使用多少核用于数据读取
use_gpu=False, 是否使用gpu, lgb 需要 param中设置使用gpu
report_dir = './encode/dist_model', 模型保存位置
```

使用样例
``` python
ds, categorical_features, numerical_features = FE.dist_data_processing("experiment/credit/credit.csv", 'class')

ME.dist_model(ds, 'class', categorical_features=categorical_features)
```


## 完整流程案例-常规
``` python

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
```


## 完整流程案例-分布式
``` python

import ray
import pandas as pd

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
```
