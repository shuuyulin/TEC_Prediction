import pandas as pd
import statistics

train_df = pd.read_csv('./single_point_train.csv')

min_max_p = {}
z_score_p = {}
for idx, col in enumerate(train_df):
    if idx in range(3):
        continue
    min_max_p[col] = (min(train_df[col]), max(train_df[col]))
    z_score_p[col] = (statistics.mean(train_df[col]), statistics.variance(train_df[col]))
    
print(min_max_p)
print(z_score_p)

import json

json.dump(min_max_p, open('./min_max_p.json', 'w'))
json.dump(z_score_p, open('./z_score_p.json', 'w'))
    