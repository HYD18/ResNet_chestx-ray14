import os
import pandas as pd

log_dir = "/root/autodl-tmp/project/result"
model_list = [
    'resnet18_log.csv',
    'resnet34_log.csv',
    'resnet50_log.csv',
    'resnet101_log.csv',
    'resnet152_log.csv',
]

for file in model_list:
    path = os.path.join(log_dir, file)
    df = pd.read_csv(path)
    print(f"{file} 列名: {df.columns.tolist()}")