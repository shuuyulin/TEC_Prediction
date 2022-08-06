import pandas as pd
import statistics

train_df = pd.read_csv('./single_point_train.csv', header=list(range(6)), index_col=0)
renamelist = ['Year', 'Day', 'Hour', 'Kp index', 'R', 'Dst-index, nT', 'ap_index, nT', 'f10.7_index'] +\
            [(67.5, -65), (25, 120), (0, -90), (-20, -160), (-32.5, 20), (-77.5, 165)]
train_df.columns = renamelist
print(train_df.info())
min_max_p = {}
z_score_p = {}
for idx, col in enumerate(train_df):
    if idx in range(3):
        continue
    min_max_p[str(col)] = (min(train_df[col]), max(train_df[col]))
    z_score_p[str(col)] = (statistics.mean(train_df[col]), statistics.variance(train_df[col]))
    
print(min_max_p)
print(z_score_p)


import json

json.dump(min_max_p, open('./min_max_p.json', 'w'))
json.dump(z_score_p, open('./z_score_p.json', 'w'))
    