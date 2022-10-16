from distutils.command.config import config
import torch
import math
import os
import json
from torch.utils.data import Dataset
import random
from utils import *

class SWGIMDataset(Dataset):
    def __init__(self, config, df, truth_df, data_indices, processer, task="train"):
        self.config = config
        self.df = df
        self.truth_df = truth_df
        self.data_indices = data_indices
        self.processer = processer
        
        self.reduce = config.getboolean('data', 'reduce')
        self.input_time_step = config.getint('model', 'input_time_step')
        self.output_time_step = config.getint('model', 'output_time_step')
        self.predict_range = self.config['global']['predict_range']
        self.seq_base = self.config['data']['seq_base']
        self.feat_slicing, self.truth_slicing = self._get_feature_slice()
        
        if task == 'train' and self.config['train']['shuffle'] == 'True':
            random.shuffle(self.data_indices)
        else:
            self.reduce = False
        
        self.reduce_ratio = config.getfloat('data', 'reduce_ratio') if self.reduce else 1
        # self._preprocess()
        
    def _get_feature_slice(self):
        feat_idx_dict = {'kp':[3], 'r':[4], 'dst':[5], 'ap':[6], 'f10.7':[7], 'storm_state':[8],\
            'storm_size':[9], 'tec':list(range(10,10+71*72)), 'tec_sh':list(range(10+71*72,10+71*72+256))}
        if self.predict_range != 'global': feat_idx_dict['tec'] = [10]
        
        features = config2strlist(self.config['data']['features'])
        rt_slice = []
        for ft in features: rt_slice += feat_idx_dict[ft]
        
        truth_df_slice = list(range(71*72)) if features[-1] == 'tec' else list(range(71*72,71*72+256))
        return rt_slice, truth_df_slice
    
    def _preprocess(self):
              
        if self.config['preprocess']['normalization_type'] != 'None':
            self.df = self.processer.preprocess(self.df)
                        
        if self.config['preprocess']['predict_norm'] == 'True':
            self.truth_df = self.processer.preprocess(self.truth_df)
            
    def __len__(self):
        return int(self.reduce_ratio * len(self.data_indices))
        
    def __getitem__(self, idx):
        data_idx = self.data_indices[idx]

        x = self.df.iloc[data_idx:data_idx + self.input_time_step, self.feat_slicing]
        # input_time_step, feature_num
        try:
            y = self.truth_df.iloc[data_idx:data_idx+self.output_time_step, self.truth_slicing]
        except:
            raise IndexError(f'Index error {idx}, {data_idx}')
        
        x = torch.tensor(x.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)

        if self.predict_range == 'global':          
            if self.seq_base == 'latitude': # TODO: lat, long for SW, SH features
                x = torch.permute(x.view(-1, 71, 72), (1, 0, 2)).reshape(71, -1)
                # y = torch.permute(y.view(-1, 71, 72), (1, 0, 2)).reshape(71, -1)
            elif self.seq_base == 'longtitude':
                x = torch.permute(x.view(-1, 71, 72), (2, 0, 1)).reshape(72, -1)
                # y = torch.permute(y.view(-1, 71, 72), (2, 0, 1)).reshape(72, -1)
                
        return  x, y
