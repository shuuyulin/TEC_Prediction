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
        self.seq_pos_feature = self.config.getboolean('data', 'seq_feature')
        self.date_slicing, self.glob_slicing, self.tec_slicing, self.truth_slicing = self._get_feature_slice()
        self.notHibert = not (self.config['data']['date_seq_base_norm'] == "Hibert")

        if task == 'train' and self.config['train']['shuffle'] == 'True':
            random.shuffle(self.data_indices)
        else:
            self.reduce = False
        
        self.reduce_ratio = config.getfloat('data', 'reduce_ratio') if self.reduce else 1
        # self._preprocess()
        
    def _get_feature_slice(self):
        feat_idx_dict = {**{ft:[idx] for idx, ft in enumerate(['year', 'DOY', 'hour', 'kp', 'r', 'dst', 'ap', 'f10.7', 'storm_state',\
            'storm_size'])}, 'tec':list(range(10,10+71*72)), 'tec_sh':list(range(10+71*72,10+71*72+256)), 'latitude':[], 'longitude':[]}
        if self.predict_range != 'global': feat_idx_dict['tec'] = [10]
        
        date_features = config2strlist(self.config['data']['date_features'])
        global_features = config2strlist(self.config['data']['global_features'])
        tec_features = self.config['data']['tec_features']
        
        date_slice, glob_slice = [], []
        for ft in date_features: date_slice += feat_idx_dict[ft]
        for ft in global_features: glob_slice += feat_idx_dict[ft]
        tec_slice = feat_idx_dict[tec_features]
        
        truth_df_slice = list(range(71*72)) if tec_features == 'tec' else list(range(71*72,71*72+256)) # TODO: single point
        return date_slice, glob_slice, tec_slice, truth_df_slice
    
    def _preprocess(self):
              
        if self.config['preprocess']['normalization_type'] != 'None':
            self.df = self.processer.preprocess(self.df)
                        
        if self.config['preprocess']['predict_norm'] == 'True':
            self.truth_df = self.processer.preprocess(self.truth_df)
                
    def __len__(self):
        return int(self.reduce_ratio * len(self.data_indices))
    
    def __getitem__(self, idx):
        data_idx = self.data_indices[idx]

        time = self.df.iloc[data_idx+self.output_time_step, self.date_slicing] # return target time
        glob = self.df.iloc[data_idx:data_idx + self.input_time_step, self.glob_slicing] # empty slicing return empty df but same shape
        tec = self.df.iloc[data_idx:data_idx + self.input_time_step, self.tec_slicing]
        y = self.truth_df.iloc[data_idx:data_idx+self.output_time_step, self.truth_slicing]
                
        # input_time_step, feature_num
        time = torch.tensor(time.values, dtype=torch.float32)
        glob = torch.tensor(glob.values, dtype=torch.float32)
        tec = torch.tensor(tec.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)

        # mapping to hibert space
        time = torch.cat((mapping2Hibert(time[0:1], 366, self.notHibert), mapping2Hibert(time[1:2], 24, self.notHibert)), dim=-1) # ERROR: time index out of range
        
        x_list = []
        if self.predict_range != 'global':
            x_list = [tec, glob]
        else: # global prediction
            if self.seq_base == 'time':
                x_list = [tec, glob]
            if self.seq_base == 'latitude': # TODO: SH tec
                tec = torch.permute(tec.view(-1, 71, 72), (1, 0, 2)).reshape(71, -1)
                glob = torch.cat((glob.reshape(-1), time), dim=0).repeat(71, 1)
                seq_pos = mapping2Hibert((torch.linspace(0, 71, 71)).unsqueeze(1), 71, self.notHibert) if self.seq_pos_feature else torch.tensor([])
                x_list = [tec, glob, seq_pos]
                    
            elif self.seq_base == 'longitude':
                tec = torch.permute(tec.view(-1, 71, 72), (2, 0, 1)).reshape(72, -1)
                glob = torch.cat((glob.reshape(-1), time), dim=0).repeat(72, 1)
                seq_pos = mapping2Hibert((torch.linspace(0, 72, 72)).unsqueeze(1), 72, self.notHibert) if self.seq_pos_feature else torch.tensor([])
                x_list = [tec, glob, seq_pos]
        # for i in x_list:
        #     print(i.shape)
        
        x = torch.cat(x_list, dim=1)
        return  x, y
