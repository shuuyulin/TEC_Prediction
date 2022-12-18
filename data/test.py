import pandas as pd
import statistics
from pathlib import Path
from tqdm import tqdm

def read_csv_data(config, mode, DATAPATH):
    years = [int(y) for y in config['data'][f'{mode}_year'].split(',')]
    
    pred_range = config['global']['predict_range']
    # if not global drop other location
    if pred_range not in ['global', 'globalSH']:
        # lng, lat = config2strlist(config['global']['predict_range'])
        
        # all_df = pd.read_csv(DATAPATH / Path(f'single_point_{mode}.csv'), header=list(range(6)), index_col=0)
        # # drop columns
        # use_cols = list(range(8)) + [list(all_df.columns).index(('CODE', 'GIM', '10TEC', 'I3', lng, lat))]
        # all_df = all_df.iloc[:, use_cols]
        pass
    
    else:
        use_cols = list(range(10 + 71*72)) + list(range(10 + 71*72*2, 10 + 71*72*2 + 256))
        # droplist = [0, 9, 10] + list(range(5122, 10235)) # 71*72 + 10 = 5122
        # renamelist = ['year', 'DOY', 'hour', 'Kp index', 'R', 'Dst-index, nT', 'ap_index, nT', 'f10.7_index'] +\
        #                 [(lat*2.5, lng) for lat in range(35, -36, -1) for lng in range(-180, 180, 5)]

        df_list = []
        print('Reading csv data...')
        for year in tqdm(years):
            year_df = pd.read_csv(DATAPATH / Path(f'raw_data/SWGIM_year/{year}.csv'),\
                header=list(range(6)), index_col=0)
            
            year_df = year_df.iloc[:, use_cols]
            # rename dataframe
            # year_df.columns = renamelist
            
            df_list.append(year_df)
            
        all_df = pd.concat(df_list, axis=0)
    
              
    return all_df

config = {'global':{'predict_range':'globalSH'},'model':{'input_time_step':24}
          ,'data':{'train_year':'2018, 2019'}}
train_df = read_csv_data(config, 'train', './')


min_max_p = {}
z_score_p = {}
for idx, col in enumerate(train_df):
    # if idx in range(3):
    #     continue
    min_max_p[str(col)] = (min(train_df[col]), max(train_df[col]))
    z_score_p[str(col)] = (statistics.mean(train_df[col]), statistics.variance(train_df[col]))
    
# print(min_max_p)
# print(z_score_p)


import json

json.dump(min_max_p, open('./min_max_p.json', 'w'))
json.dump(z_score_p, open('./z_score_p.json', 'w'))
    