import torch
import math
import os
import json
from torch.utils.data import Dataset
import random
from utils import *

class SWGIMDataset(Dataset):
    def __init__(self, config, df, data_indices, task="train"):
        self.config = config
        self.df = df
        self.data_indices = data_indices
        
        self.reduce = config.getboolean('data', 'reduce')
        self.input_time_step = config.getint('model', 'input_time_step')
        self.output_time_step = config.getint('model', 'output_time_step')
        
        self.predict_range = config['global']['predict_range']
        if self.predict_range != 'global':
            self.predict_range = configlist2intlist(self.predict_range)
        
        # # apply normalization to space weather and tec
        # self.df[self.df.columns[3: len(self.df.columns)]] = norm(self.df[self.df.columns[3: len(self.df.columns)]])
        
        if task == 'train':
            random.shuffle(self.data_indices)
        else:
            self.reduce = False
        
        self.reduce_ratio = config.getfloat('data', 'reduce_ratio') if self.reduce else 1

    def __len__(self):
        return int(self.reduce_ratio * len(self.data_indices)) if self.reduce else len(self.data_indices)
        # ==ISSUE== Should start of data (input len < input_time_step) be included? (currently not included)
        
    def __getitem__(self, idx):
        data_idx = self.data_indices[idx]

        space_data = self.df.iloc[data_idx:data_idx + self.input_time_step, 3:8]
        tec_data = self.df.iloc[data_idx:data_idx + self.input_time_step, 8:len(self.df.columns)]
        tec_truth = self.df.iloc[data_idx + self.input_time_step + self.output_time_step - 1, 8:len(self.df.columns)]
            
        return  torch.tensor(space_data.values), torch.tensor(tec_data.values),\
                torch.tensor(tec_truth.values)

