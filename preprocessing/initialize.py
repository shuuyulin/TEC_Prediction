from .normalization import MinMaxNorm, StandardNorm
import json
import pandas as pd
import torch
def initialize_processer(config, *args, **kwargs):
    
    normalization_type_list = {
        'min_max' : MinMaxNorm,
        'z_score' : StandardNorm,
        'None' : None,
    }
    
    normalization_type = config['preprocess']['normalization_type']
    
    if normalization_type == 'None':
        return None
    
    if normalization_type in normalization_type_list:
        norm = normalization_type_list[normalization_type]()
    else:
        print('normalization_type has not been defined in config file!')
        raise AttributeError
    
    norm_params = json.load(open(f'./data/{normalization_type}_p.json', 'r'))
    
    return processer(norm, norm_params)
        
class processer():
    def __init__(self, norm, norm_params):
        self.norm = norm
        self.norm_params = norm_params
        self.tec_norm_params = [d for d in self.norm_params][10:10+5112]
        # self.norm_params = [0, 400] # TODO: test all loc using same params
    def preprocess(self, df:pd.DataFrame):
        
        for col in df.columns:
            param_key = str(col)     
            if col[1] in ['year','DOY', 'hour']:#('year','DOY','hour', 'Geomagnetic Storms Size','Geomagnetic Storms State'):
                pass
            elif param_key in self.norm_params:
                df[col] = self.norm.normalize(df[col], *self.norm_params[param_key])
                # df[col] = self.norm.normalize(df[col], *self.norm_params)
            else:
                print(f'key {param_key} not exist in norm_params, ignored')
                raise KeyError
        return df
    def preprocess_t(self, tensor:torch.tensor):
        for idx, param_key in enumerate(self.tec_norm_params):
            # print(tensor.shape)
            # print(self.norm_params[param_key])
            tensor[..., idx] = self.norm.normalize(tensor[..., idx], *self.norm_params[param_key])
        return tensor

    def postprocess(self, df:pd.DataFrame):
        for col in df.columns:
            param_key = str(col)     
            if col[1] in ['year','DOY', 'hour']:#('year','DOY','hour', 'Geomagnetic Storms Size','Geomagnetic Storms State'):
                pass
            elif param_key in self.norm_params:
                df[col] = self.norm.denormalize(df[col], *self.norm_params[param_key])
                # df[col] = self.norm.normalize(df[col], *self.norm_params)
            else:
                print(f'key {param_key} not exist in norm_params, ignored')
                raise KeyError
        return df
        
        