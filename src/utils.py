import random
import numpy as np
import pandas as pd
import torch
import os
from pathlib import Path
from tqdm.auto import tqdm
from pathlib import Path
import logging

def setSeed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    pd.core.common.random_state(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
  
def config2intlist(confstr) -> list:
    return [int(i) for i in confstr.split(',') if i != '']
def config2strlist(confstr) -> list:
    return [i.strip() for i in confstr.split(',') if i != '']
def config2floatlist(confstr) -> list:
    return [float(i) for i in confstr.split(',') if i != '']
def config2boollist(confstr) -> list:
    return [i == "True" for i in confstr.split(',') if i != '']

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
            logging.warning(f'Warning: replacing folder {RECORDPATH}')
        logging.info(f'Creating folder: {RECORDPATH}')
        RECORDPATH.mkdir(parents=True, exist_ok=True)
    else: # test
        RECORDPATH = Path(args.record)
        
    return RECORDPATH
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
