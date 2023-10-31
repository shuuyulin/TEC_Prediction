import torch
from torch.utils.data import Dataset
import random
from ..utils import *

class SWGIMDataset(Dataset):
    def __init__(self, config, essnse, data_indices, task="train"):
        self.config = config
        self.data = essnse['data']
        self.shaper = essnse['shaper']
        self.data_indices = data_indices
        
        self.reduce = config.getboolean('data', 'reduce')
        self.i_step = config.getint('model', 'input_time_step')
        self.o_step = config.getint('model', 'output_time_step')
        
        # if task == 'train' and self.config['train']['shuffle'] == 'True':
        #     random.shuffle(self.data_indices)
        if task != 'train':
            self.reduce = False
        
        self.reduce_ratio = config.getfloat('data', 'reduce_ratio') if self.reduce else 1
        
    def __len__(self):
        return int(self.reduce_ratio * len(self.data_indices))
    
    def __getitem__(self, idx):
        data_idx = self.data_indices[idx]
        input = {k:torch.tensor(self.data['input'][k][data_idx:data_idx + self.i_step], dtype=torch.float32)
                 for k in self.data['input']}
        truth = {k:torch.tensor(self.data['truth'][k][data_idx + self.i_step:data_idx + self.i_step + self.o_step], dtype=torch.float32)
                 for k in self.data['truth']}
        
        logging.debug(f'tec shape: {input["tec"].shape}')
        if input['tec'] is not None:
            input['tec'] = self.shaper.shape_input(input['tec'])
        
        seq_len = input['tec'].shape[0]
        for k in input:
            if k in ['year','DOY','hour','kp', 'r', 'dst', 'ap', 'f10.7', 'storm_state', 'storm_size']:
                input[k] = input[k].reshape(-1).repeat(seq_len, 1)

        x = torch.cat(list(input.values()), dim=1)
        return x, truth #TODO possible error, truth is dictionary
