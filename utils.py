import random
import numpy as np
import pandas as pd
import torch
import os
from pathlib import Path
from tqdm.auto import tqdm
from pathlib import Path

def setSeed(seed=31):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    pd.core.common.random_state(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
    """
    Read SWGIM data from path
    and rename, drop columns
    """

def read_csv_data(config, mode, DATAPATH):
    years = [int(y) for y in config['data'][f'{mode}_year'].split(',')]
    
    pred_range = config['global']['predict_range']
    # if not global drop other location
    if pred_range not in ['global', 'globalSH']:
        lng, lat = config2strlist(config['global']['predict_range'])
        
        all_df = pd.read_csv(DATAPATH / Path(f'single_point_{mode}.csv'), header=list(range(6)), index_col=0)
        # drop columns
        use_cols = list(range(8)) + [list(all_df.columns).index(('CODE', 'GIM', '10TEC', 'I3', lng, lat))]
        # all_df = all_df.iloc[:, use_cols]
        all_df = all_df.loc[all_df.columns[:8] + [('CODE', 'GIM', '10TEC', 'I3', lng, lat)]]
        
    
    else:
        use_cols = list(range(8))
        use_cols += list(range(10, 10 + 71*72)) if pred_range == 'global'\
            else list(range(10 + 71*72*2, 10 + 71*72*2 + 256))
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
            
        all_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    
    
    # get truth_df
    tmp = int(config['model']['input_time_step'])
    truth_df = all_df.drop(columns='OMNIWeb').iloc[tmp:].reset_index(drop=True)
    del tmp
          
    return all_df, truth_df

def get_indices(config, all_df, seed, mode='train', p=0.8):
    """return indices of train, valid data / test data

    Args:
        all_df (Dataframe): data to be splited
        seed (int): random seed
        p (float, optional): Spliting ratio. Defaults to 0.8.

    Returns:
        indices (tuple): (train_indices, valid_indices)
            delete data indices which total needed data exceed df
            shuffled 
    """
    p = config.getfloat('data', 'valid_ratio')
    
    tc_limit = config.getint('model', 'input_time_step') + config.getint('model', 'output_time_step')
    reserved = config.getint('data', 'reserved')
    
    indices = all_df.index[max(0, reserved - tc_limit + 1):len(all_df.index) - tc_limit + 1].to_series()
    if mode == 'train':
        indices = indices.sample(frac=1, random_state=seed).tolist()
        return indices[:int(len(indices)*p)], indices[int(len(indices)*p):]   
    elif mode == 'test':
        return indices.tolist()
  
def config2intlist(confstr) -> list:
    return [int(i) for i in confstr.split(',')]
def config2strlist(confstr) -> list:
    return [i.strip() for i in confstr.split(',')]

import matplotlib.pyplot as plt
def plot_fg(x1, title, y, path, x2=None):
    plt.clf()
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(y)
    plt.plot(x1, label='train')
    if x2 is not None:
        plt.plot(x2, label='valid')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(path, title+'.jpg'))
    
import re
def get_record_path(args):
    if args.mode == 'train':
        RECORDPATH = Path(args.record)
        if RECORDPATH.exists() and any(RECORDPATH.iterdir()):
            print(f'Warning: replacing folder {RECORDPATH}')
        print(f'Creating folder: {RECORDPATH}')
        RECORDPATH.mkdir(parents=True, exist_ok=True)
    else: # test
        RECORDPATH = Path(args.record)
        
    return RECORDPATH
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
