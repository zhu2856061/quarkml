# -*- coding: utf-8 -*-
# @Time   : 2021/10/27 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
import setuptools

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'pandas>=1.2.1', 
    'pandas-profiling>=3.6.6',
    'lightgbm>=3.3.3',
    'scikit-learn>=1.0.2',
    'xgboost>=1.6.2',
    'cython>=0.29.32',
    'joblib>=1.2.0',
    'loguru>=0.7.0',
    'matplotlib>=3.5.1',
    'seaborn>=0.12.2',
    'featurewiz>=0.3.2',
    'shap>=0.41.0',
    'tqdm>=4.65.0',
    'scipy>=1.6.0',
    'hyperopt>=0.2.7',
    'optuna>=3.3.0',
    'ray>=2.6.2',
    'ray[default]',
    'lightgbm_ray>=0.1.8',
    'treelite>=3.0.0',
    'treelite_runtime>=3.0.0',
]
setuptools.setup(
    name="quarkml",
    version="0.0.1",
    author="merlin",
    description="easy-to-use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="None",
    packages=setuptools.find_packages(exclude=["bak", "experiment"]),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=REQUIRED_PACKAGES,
    entry_points={},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license="Apache-2.0",
    keywords=['train framework', 'xgb'],
)