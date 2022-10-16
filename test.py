import torch
import torch.nn as nn
import pandas as pd

df = pd.read_csv('./record/39/prediction.csv', header=list(range(6)), index_col=0)

print(df.head())