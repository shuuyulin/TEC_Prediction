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
    
def read_csv_data(config, mode, DATAPATH):
    """
    Read SWGIM data from path
    and drop columns
    Returns:
        all_df (DataFrame):
            All Data as pandas DataFrame data format
        truth_df (DataFrame):
            Truth data as pandas DataFrame data format
    """
    years = [int(y) for y in config['data'][f'{mode}_year'].split(',')]
    
    pred_range = config['global']['predict_range']
    
    if pred_range != 'global': # single point prediction, drop other location
        lng, lat = config2strlist(config['global']['predict_range'])
        
        all_df = pd.read_csv(DATAPATH / Path(f'single_point_{mode}.csv'), header=list(range(6)), index_col=0)
        # drop columns
        all_df = all_df.loc[all_df.columns[:8] + [('CODE', 'GIM', '10TEC', 'I3', lng, lat)]]
            
    else: # global
        use_cols = list(range(10 + 71*72)) + list(range(10 + 71*72*2, 10 + 71*72*2 + 256))
        # drop TEC error, sh error

        df_list = []
        print('Reading csv data...')
        for year in tqdm(years, dynamic_ncols=True):
            year_df = pd.read_csv(DATAPATH / Path(f'raw_data/SWGIM_year/{year}.csv'),\
                header=list(range(6)), index_col=0)
            
            year_df = year_df.iloc[:, use_cols]
            # rename dataframe
            # year_df.columns = renamelist
            
            df_list.append(year_df)
            
        all_df = pd.concat(df_list, axis=0).reset_index(drop=True)
        
    # get truth_df
    tmp = int(config['model']['input_time_step'])
    truth_df = all_df.iloc[tmp:, 10:]
    # drop date, SW and label
    del tmp
          
    return all_df, truth_df

def get_indices(config, all_df, seed, mode='train', p=0.8):
    """return indices of train, valid data / test data

    Args:
        all_df (Dataframe): data to be splited
        seed (int): random seed
        p (float, optional): Spliting ratio. Defaults to 0.8.

    Returns:
        indices (tuple): (valid_indices, train_indices)
            delete data indices which total needed data exceed df
            shuffled 
    """
    p = config.getfloat('data', 'valid_ratio')
    
    i_step, o_step = int(config['model']['input_time_step']), int(config['model']['output_time_step'])
            
    if mode == 'train':
        indices = all_df.index[:len(all_df.index) - i_step - o_step + 1].to_series()
        indices = indices.sample(frac=1, random_state=seed).tolist()
        # indices = indices.tolist()
        return indices[:int(len(indices)*p)], indices[int(len(indices)*p):]   
    elif mode == 'test':
        # k = len(all_df.index) - i_step - (0 if config['model']['model_name'] == 'Transformer_ED' else o_step - 1)
        k = len(all_df.index) - i_step - o_step + 1
        indices = all_df.index[:k].to_series()
        return indices.tolist()
  
def config2intlist(confstr) -> list:
    return [int(i) for i in confstr.split(',') if i != '']
def config2strlist(confstr) -> list:
    return [i.strip() for i in confstr.split(',') if i != '']

def mapping2Hibert(value, limit=1, N=False):
    # FLAG: feature normalization min-max / sin-cos
    if N:
        return value    
    return torch.cat((torch.sin(value / limit * 2 * np.pi), torch.cos(value / limit * 2 * np.pi)), dim=-1)

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
