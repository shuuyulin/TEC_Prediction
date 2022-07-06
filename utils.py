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
    
    # if not global drop other location
    if config['global']['predict_range'] != 'global':
        lat, lng = configlist2intlist(config['global']['predict_range'])
        
        droplist = [0] + list(range(9, 10235))
        droplist.remove(10 + int((87.5-lat)/2.5)*72 + int((180+lng)/5+1))
        
        renamelist = ['Year', 'Day', 'Hour', 'Kp index',
              'R', 'Dst-index, nT', 'ap_index, nT', 'f10.7_index', (lat, lng)]
    else: # ==TODO== check if global renamelist error
        droplist = [0, 9, 10] + list(range(5122, 10235)) # 71*72 + 10 = 5122
        renamelist = ['Year', 'Day', 'Hour', 'Kp index', 'R', 'Dst-index, nT', 'ap_index, nT', 'f10.7_index'] +\
                        [(lat*2.5, lng) for lat in range(35, -36, -1) for lng in range(-180, 180, 5)]

    df_list = []
    print('Reading csv data...')
    for year in tqdm(years):
        year_df = pd.read_csv(DATAPATH / Path(f'{year}.csv'), header=list(range(6)))

        # drop columns
        year_df.drop(year_df.columns[droplist], inplace=True, axis=1, errors='ignore')
        
        # rename dataframe
        year_df.columns = renamelist
        
        df_list.append(year_df)
        
    all_df = pd.concat(df_list, axis=0)
        
    return all_df

def get_indices(config, all_df, seed, mode='train', p=0.8):
    """return indices of train, valid data

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
    
    in_tc, out_tc = config.getint('model', 'input_time_step'), config.getint('model', 'output_time_step')
    
    indices = all_df.index.to_series().iloc[:len(all_df.index) - in_tc - out_tc + 1]
    if mode == 'train':
        indices = indices.sample(frac=1, random_state=seed).tolist()
        return indices[:int(len(indices)*p)], indices[int(len(indices)*p):]   
    elif mode == 'test':
        return indices
  
def configlist2intlist(confstr) -> list:
    return [int(i) for i in confstr.split(',')]

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
def get_record_path(RECORDPATH, args):
    if args.mode == 'train': # ==ISSUE== error creating new record folder
        cnt = max([int(p) if re.match('\d+', str(p)) else 0 for p in RECORDPATH.iterdir()], default=0) + 1
        RECORDPATH = RECORDPATH / Path(str(cnt))
        print('Creating folder: {RECORDPATH}')
        RECORDPATH.mkdir(parents=True, exist_ok=True)
    else: # test
        RECORDPATH = Path(args.model_path).parent
        
    return RECORDPATH
    
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
