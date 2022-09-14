import torch
import math
import os
import json
from torch.utils.data import Dataset
import random
from utils import *

class SWGIMDataset(Dataset):
    def __init__(self, config, df, truth_df, data_indices, task="train"):
        self.config = config
        self.df = df
        self.truth_df = truth_df
        self.data_indices = data_indices
        
        self.reduce = config.getboolean('data', 'reduce')
        self.input_time_step = config.getint('model', 'input_time_step')
        self.output_time_step = config.getint('model', 'output_time_step')
        self.predict_range = config['global']['predict_range']
        if self.predict_range not in ['global', 'globalSH']:
            self.predict_range = config2intlist(self.predict_range)
        
        if task == 'train':
            random.shuffle(self.data_indices)
        else:
            self.reduce = False
        
        self.reduce_ratio = config.getfloat('data', 'reduce_ratio') if self.reduce else 1

    def __len__(self):
        return int(self.reduce_ratio * len(self.data_indices)) if self.reduce else len(self.data_indices)
        
    def __getitem__(self, idx):
        data_idx = self.data_indices[idx]

        space_data = self.df.iloc[data_idx:data_idx + self.input_time_step, 3:8]
        tec_data = self.df.iloc[data_idx:data_idx + self.input_time_step, 8:]
        try:
            tec_truth =\
                self.truth_df.iloc[data_idx:data_idx+self.output_time_step, 3:]
        except:
            raise IndexError(f'Index error {idx}, {data_idx}')
            
        return  torch.tensor(space_data.values, dtype=torch.float32),\
                torch.tensor(tec_data.values, dtype=torch.float32),\
                torch.tensor(tec_truth.values, dtype=torch.float32)

