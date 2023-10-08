#!/bin/bash

# 启动ray - 本地模式
ray start --head --port=1063 --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265

# 启动ray - 集群模式
# ray start --address='10.13.26.25:1063' # 10.13.26.25 是主节点

# jupyter
jupyter lab --NotebookApp.notebook_dir='/home/quarkml' --allow-root  --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.password=''  --port=8888

# fkask
# python /home/server.py 

