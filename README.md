# quarkml

## 介绍

适用于风控算法领域，集 特征处理，特征工程，模型训练， 树模型分布式训练等功能于一体的自动化工具
主要功能：
    1. 特征工程
       a. 特征处理，分类特征转换数值index，空值填充，异常值去除, 数据压缩，特征分析报告打印
       b. 特征衍生，基于内置的算子进行自动衍生，并基于booster方法进行筛选海量衍生特征
       c. 特征选择，对自特征进行筛选【fwiz，iv，psi，tmodel】
    2. 模型工程
       a. 模型交叉验证，用于验证模型性能
       b. 模型超参
       c. 模型训练，提供树模型多进程训练，分布式训练，常规训练
       d. 模型解释性，自动化产生shap 模型解释图
       e. 预估加速，对于常规进行joblib保存的模型进行编译，生成编译文件，然后利用load_model，predict进行预估
    3. 分布式工程
       a. 分布式训练lightgbm
       b. 分布式数据处理

### 特征工程 FeatureEngineering 功能介绍

#### 1 数据预处理

##### 1.1 功能

1. 划分数据集中的类别特征和数值特征
   其中会根据数据中数值特征的唯一值总量进行判断，若总量小于ordinal_number（可配置参数），会划入类别特征
2. 缺失值填充
   对类别特征的填充采用数据中的众数，对数值特征的填充采用数据中的均值
   会保留每个特征的填充逻辑，方便后续新数据复用
3. 异常值删除
   检测异常的方法一：均方差
   在统计学中，如果一个数据分布近似正态，
   那么大约 68% 的数据值会在均值的一个标准差范围内，
   大约 95% 会在两个标准差范围内，
   大约 99.7% 会在三个标准差范围内。
   这里采用3个标准差以外的进行去除

4. 对离散特征进行数值化，tokenizer
   对每列特征进行转换index, [男， 女] -> [0, 1] 并保留{男: 0, 女: 1}
   会保留每个特征的tokenizer逻辑，方便后续新数据复用

5. 展示数据中的每个 feature-label 分布
   会展示数据的基础信息和基本描述
   会有每个特征与目标的分布，保存图到本地
   会给出整数据的详细分析报告，保存为html形式，可直接打开查看

6. 对数据进行压缩
   对数据中的数值化数据进行分析，判断数据的中最大最小范围，然后对数据采用合理的数据类型【int64-int32-int16-float64-float32-float16】等

##### 1.2 API

1. data_processing_fit()

1.1 描述

```python
data_processing_fit(
  ds: str | pd.DataFrame,
  label: str,
  cat_feature: List = [],
  num_feature: List = [],
  ordinal_number=100,
  is_fillna=False,
  drop_outliers=False,
  is_token=True,
  verbosity=False,
  compress=False,
  report_dir="./encode",
)
ds: str | pd.DataFrame, 原始数据 （必选项）, 若传入文件路径，则返回处理后的新文件路径（处理后的数据文件）， 若传入DataFrame，则返回新的DataFrame
label: str,  原始数据的label （必选项）
cat_feature: list = [], 指定类别特征， （非选项）， 若为空，则会利用上述【功能1】获得
num_feature: list = [], 指定连续特征， （非选项）， 若为空，则会利用上述【功能1】获得
ordinal_number=100, 指定数值特征若唯一值总数小于ordinal_number，则会划分成类别特征，用于【功能1】， （非选项）， 若为空，默认 100
is_fillna=False, 是否自动缺失值填充【功能2】，连续值填充众数，离散值填充均值
drop_outliers=False, 是否异常值删除【功能3】
is_token=True, 是否数值化【功能4】
verbosity=False, 是否展示分析报告【功能5】
compress=False, 是否对数据进行压缩【功能6】
report_dir="./encode", 上述功能产生的中间结果都会落到该文件夹内
```

1.2 使用样例(所有示例均可在experiment中运行测试)

```python
from quarkml.feature_engineering import FeatureEngineering
FE = FeatureEngineering()
# 直接文件路径模式
ds, cat, con = FE.data_processing_fit("credit.csv", 'class')

# dataframe 模式
ds = pd.read_csv("credit.csv")
ds, cat, con = FE.data_processing_fit(ds, 'class')

```

#### 2 数据预处理-新数据转换

与上述的数据预处理配套使用

##### 2.1 功能

1. 利用上述数据预处理的中间结果对新的数据进行转换，能够实现与上述数据预处理逻辑一致（有利于预估使用）

##### 2.2 API

1. data_processing_transform()

1.1 描述

```python
data_processing_transform(
  ds: pd.DataFrame,
  label: str,
  verbosity=False,
  compress=False,
  report_dir="./encode",
)
ds: str | pd.DataFrame, 原始数据 （必选项）, 若传入文件路径，则返回处理后的新文件路径（处理后的数据文件）， 若传入DataFrame，则返回新的DataFrame
label: str,  原始数据的label （必选项）
verbosity=False, 是否展示分析报告【功能5】
compress=False, 是否对数据进行压缩【功能6】
report_dir="./encode", 利用数据预处理的中间结果进行一致性转换数据
```

1.2 使用样例(所有示例均可在experiment中运行测试)

```python
from quarkml.feature_engineering import FeatureEngineering
FE = FeatureEngineering()

# 直接文件路径模式
ds, cat, con = FE.data_processing_transform("credit.csv", 'class')

# dataframe 模式
ds = pd.read_csv("credit.csv")
ds, cat, con = FE.data_processing_transform(ds, 'class')
```

#### 3 特征衍生

##### 3.1 功能

1. 对给予数据进行特征衍生，产生大量的候选特征集
2. 对衍生后的候选特征集进行筛选，获得最终对效果指标有意义的特征集

##### 3.2 API

1. feature_generation()

1.1 描述

```python
feature_generation(
   ds: str | pd.DataFrame,
   label: str,
   cat_features: List = None,
   is_filter=True,
   params=None,
   select_method='predictive',
   min_candidate_features=200,
   blocks=5,
   ratio=0.5,
   distributed_and_multiprocess=-1,
   report_dir="encode",
)

ds: str | pd.DataFrame, 原始数据 （必选项）, 若传入文件路径，则返回处理后的新文件路径（处理后的数据文件）， 若传入DataFrame，则返回新的DataFrame
label: str,  原始数据的label （必选项）
cat_features: list = None, 指定类别特征， （非选项）， 若为空，则会利用原始数据的非number类型 设置 为cat
is_filter: 是否对衍生后的特征进行筛选，由于衍生的特征数巨大，所以默认为True
params: 特征筛选过程中树模型的参数（非选项）， 若为空，有默认参数
select_method: 筛选的评估指标，predictive -> 提升收益，corr 为与目标的相关度
min_candidate_features: 最小候选特征，若每个block后，候选特征小于最小候选特征，就停止block了
blocks： 主要是用于booster 中对数据进行增量式划分成多个数据块
ratio：主要是用于booster 中candidate_features被淘汰ratio=0.5 一半了就，可以直接全量数据进行淘汰了，数据量越大，该值应该设置越小一般在0.2 ~ 0.5
distributed_and_multiprocess=-1， 三种运行模式【常规 -1，多进程 2，分布式 1】
report_dir="./encode", 中间结果存储
```

1.2 使用样例(所有示例均可在experiment中运行测试)

``` python
# 直接文件路径模式
ds = FE.feature_generation("credit.csv", 'class', is_filter=True)
# dataframe 模式
ds = FE.feature_generation(ds, 'class', cat, is_filter=True)
```

#### 4 特征选择

##### 4.1 功能

1. 对提供的数据集进行计算特征的IV值，并基于IV的过滤条件去掉某些特征
2. 对提供的数据集进行计算特征的PSI值，并基于PSI的过滤条件去掉某些特征
3. 对提供的数据集进行计算特征的fwiz值（最大最小相关性），并基于fwiz的过滤条件去掉某些特征
4. 对提供的数据集进行计算特征的重要性（tmodel），并基于特征的重要性去掉某些特征

具备4中特征选择方法【fwiz，iv，psi，tmodel】，其中除fwiz 方法外，其他方法均具备三种运行模式【常规，多进程，分布式】
小数据量下运行耗时H比较 H(多进程) < H(常规) < H(分布式)
大数据量下单机无法运行时 - 推荐用分布式

fwiz 基于SULOV（搜索不相关的变量列表），SULOV 注定只能针对连续值，
SULOV算法基于本文中解释的最小冗余最大相关性（MRMR）算法，该算法是最佳特征选择方法之一

iv 风控领域常使用的IV值

psi 风控领域常使用的PSI值

tmodel 基于树模型训练后的特征重要性进行选择，其中有4种特征重要性【importance， permutation， shap， all】
    - importance 为树的划分节点gain值总和
    - permutation 评估特征的随机和非随机的预估值差值比较（模型无关性）
    - shap 类似permutation， 只是更加严格，【具体可学习Shap 值】
    - all 是综合上述三种方法选出的特征交集

##### 4.2 API

1. feature_selector()

1.1 描述

```python
feature_selector(
ds: str | pd.DataFrame,
label: str,
part_column: str = None,
cate_features: List[str] = None,
part_values: List = None,
bins=10,
importance_metric: str = "importance",
method: str = "fwiz",
distributed_and_multiprocess=-1,
report_dir="encode",
)

ds: str | pd.DataFrame, 原始数据 （必选项）, 若传入文件路径，则返回处理后的新文件路径（处理后的数据文件）， 若传入DataFrame，则返回新的DataFrame
label: str,  原始数据的label （必选项）
cat_features: list = None, 指定类别特征， （非必选项）， 若为空，则会利用原始数据的非number类型 设置 为cat
part_column: str = None,  划分列，主要是用于PSI筛选特征的方法内，表明用这个一列进行划分数据集，然后比较每个数据两两之间的差异，当method为psi 时是必选项
part_values: str = None,  划分列，与part_column 一起用，表明用这个一列按part_values list 内的值进行划分，然后比较每个数据两两之间的差异（非必选项）， 若为空，将去part_column列中的所有值
bins=10,: IV计算中的分桶数
importance_metric: str = 'importance',  tmodel_method方法得到的重要性分为3种方式的 ： importance， permutation， shap
method: str = "booster", 特征筛选的方法：fwiz ， iv ， psi ， tmodel ， 注fwiz-基于SULOV（搜索不相关的变量列表）
distributed_and_multiprocess=-1， 三种运行模式【常规 -1，多进程 2，分布式 1】
report_dir="encode", 特征筛选过程中的一些中间结果存在路径
```

1.2 使用样例(所有示例均可在experiment中运行测试)

``` python
ds = FE.feature_selector(ds, 'class', method='fwiz')
ds = FE.feature_selector(ds, 'class', cate_features=cat,  method='iv')
ds = FE.feature_selector(ds, 'class', part_column='age', cate_features=cat,  method='psi')
ds = FE.feature_selector(ds, 'class', cate_features=cat,  method='tmodel')
```

### 模型工程 ModelEngineering

(树模型-lightgbm)
a. 模型超参
b. 模型交叉验证，用于验证模型性能
c. 模型训练，提供树模型多进程训练，分布式训练，常规训练
d. 模型解释性，自动化产生shap 模型解释图

#### 1 模型超参 hparams

##### 1.1 功能

基于提供的数据进行模型lightgbm，参数寻优，基于贝叶斯超参方式找到最优参数

##### 1.2 API

1. hparams()

1.1 描述

``` python
hparams(
   ds: pd.DataFrame,
   label: str,
   valid_ds: pd.DataFrame = None,
   cat_features=None,
   params=None,
   spaces=None,
   report_dir="encode")

ds: str | pd.DataFrame, 原始数据 （必选项）, 可以是DataFrame 或者 文件路径
label: str,  原始数据的label （必选项）
valid_ds: str | pd.DataFrame, 原始验证数据 （非必选项）, 可以是DataFrame 或者 文件路径
cat_features: list = None, 指定类别特征， （非必选项）， 若为空，则会利用原始数据的非number类型 设置 为cat
params=None, lgb模型的参数，非必选项，若没有，会采用系统默认的
spaces=None, 寻参的设置空间， 熟悉hyperopt应该知道怎么配置
report_dir="encode"
```

使用样例

``` python
best_params_hyperopt = ME.hparams(ds, 'class', cat_features=cat)
```

#### 2 模型交叉验证 model_cv

##### 2.1 功能

基于提供的数据进行lightgbm的交叉验证，能够进行分布式训练加快验证流程

##### 2.2 API

1. model_cv()

``` python

model_cv(
   ds: pd.DataFrame,
   label: str,
   valid_ds: pd.DataFrame = None,
   categorical_features=None,
   params=None,
   folds=5,
   distributed_and_multiprocess=-1,
)

ds: str | pd.DataFrame, 原始数据 （必选项）, 可以是DataFrame 或者 文件路径
label: str,  原始数据的label （必选项）
valid_ds: str | pd.DataFrame, 原始验证数据 （非必选项）, 可以是DataFrame 或者 文件路径
cat_features: list = None, 指定类别特征， （非必选项）， 若为空，则会利用原始数据的非number类型 设置 为cat
params=None, lgb模型的参数，非必选项，若没有，会采用系统默认的
folds=5,交叉验证的数据划分份数，用于tmodel
distributed_and_multiprocess=-1， 三种运行模式【常规 -1，多进程 2，分布式 1】
```

使用样例

``` python
ME.model_cv(ds, 'class')
```

#### 3 模型训练 model

##### 3.1 功能

基于提供的数据进行lightgbm的训练

##### 3.2 API

``` python

model(
   ds: pd.DataFrame,
   label: str,
   valid_ds: pd.DataFrame = None,
   cat_features=None,
   params=None,
   report_dir="encode",
)

ds: str | pd.DataFrame, 原始数据 （必选项）, 可以是DataFrame 或者 文件路径
label: str,  原始数据的label （必选项）
valid_ds: str | pd.DataFrame, 原始验证数据 （非必选项）, 可以是DataFrame 或者 文件路径
cat_features: list = None, 指定类别特征， （非必选项）， 若为空，则会利用原始数据的非number类型 设置 为cat
params=None, lgb模型的参数，非必选项，若没有，会采用系统默认的
report_dir="encode" lgb模型训练保存的路径report_dir下的 loan_model.pkl
```

使用样例

``` python
ME.model(ds, 'class')
```

#### 4 模型解释性 model_interpretable

##### 4.1 功能

   1. 基于训练好的模型，对模型为何对这个样本打出这样的分，进行单实例分析
   2. 基于训练好的模型，对模型为何对一个群体样本打出这样的分，进行多实例分析整体分析
   3. 给出该模型对这些样本预估出来的结果分析，给出特征重要性


##### 4.2 API

1. interpretable()

```
interpretable(
   task,
   model,
   X: pd.DataFrame,
   single_index: int = -1,
   muli_num: int = -1,
   is_importance=False,
)

task: 回归或者分类 ： 回归 -> regression   分类 -> class
model: 上述训练的模型，需要解释的模型
single_index: 单实例分析，-1 为不进行分析， single_index 为 X 数据的样本下标
muli_num: 多实例分析，-1 为不进行分析，muli_num 为 X 数据的前muli_num个样本
is_importance： 是否给出特征重要性
```

使用样例

``` python
ME.interpretable('regression', tm, X, single_index=1)
```

----
# 【新增】分布式树模型(lightgbm)训练，需对数据和模型进行改造

distributed_engineering()

1. dist_model() 对于处理后的数据csv，进行分布式训练，（数据内的数据务必自己提前处理好，即数据类型为 int float 不能有字符str）
2. dist_data_processing() 分布式特征处理，并将数据转换成分布式数据样式，懒加载模式
3. 这里推荐先采用spark将数据处理成可训练数据，或者采用上述提供的方法，即将原始数据处理成可训练的格式后，即可调用dist_model()进行训练

## 分布式训练 dist_model

```python
dist_model(
   ds: ray.data.Dataset | str,
   label: str,
   valid_ds: ray.data.Dataset | str = None,
   cat_features=None,
   params=None,
   num_workers=2,
   trainer_resources=None,
   resources_per_worker=None,
   use_gpu=False,
   delimiter=",",
   report_dir='./encode/dist_model',
)

ds: ray.data.Dataset | str, 原始数据 （必选项）, 可以是ray.data.Dataset 或者 文件路径
label: str,  原始数据的label （必选项）
valid_ds: str | pd.DataFrame, 原始验证数据 （非选项）, 可以是ray.data.Dataset 或者 文件路径
cat_features: list = None, 指定类别特征， （非选项）， 若为空，则会利用原始数据的非number类型 设置 为cat
params=None, lgb模型的参数，非必选项，若没有，会采用系统默认的
num_workers=2, 数据并发，有几个work进行分布计算
trainer_resources, 用于训练的资源配置 {"CPU": 2, "GPU": 2} cpu和gpu资源
resources_per_worker, 用于work的资源 {"CPU": 2, "GPU": 2} cpu和gpu资源
use_gpu=False， 是否使用gpu
delimiter="," 数据csv中的分隔符
report_dir 模型存储的位置
```

使用样例

``` python
DE.dist_model("experiment/credit/credit.csv_data_processing.csv", 'class')
ray.shutdown()
```

# 【新增】预估效率优化

predict_2_so(model_path) 将原始的joblib保存的文件编译成so，同目录下会生成so文件
predict_load_so(model_path) 加载模型
predict_x(x) 利用加载的模型进行预估

``` python
predict_2_so("encode/loan_model.pkl")
predict_load_so("encode/loan_model.pkl_lite")
predict_x(X)
```

---

# 环境

```
1 安装 
  pip3 install ray[default]
  pip3 install quarkml-0.0.1-py3-none-any.whl (请将整个代码进行打包成whl,进行安装，打包方式quark-ml目录下 执行 python setup.py bdist_wheel)
  切记不可 pip3 install ray，因为这样安装不完整，会没有Dashboard
2.启动
  【启动 head 节点】ray start --head --port=1063 --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265
  【启动其他节点】 ray start --address='head节点的ip:1063'

这里只演示启动head 节点，并在head节点运行，因此 ray.init() 并没有设置
```

## 完整流程案例- 参考experiment中的credit的run.py脚本
