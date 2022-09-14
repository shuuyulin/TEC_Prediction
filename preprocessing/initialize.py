from .normalization import MinMaxNorm, StandardNorm
import json
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
        # self.norm_params = [0, 400] # TODO: test all loc using same params
    def preprocess(self, df):
        
        for col in df.columns:
            param_key = str(col)     
            if col[1] in ('year','DOY','hour'):
                pass
            elif param_key in self.norm_params:
                df[col] = self.norm.normalize(df[col], *self.norm_params[param_key])
                # df[col] = self.norm.normalize(df[col], *self.norm_params)
            else:
                print(f'key {param_key} not exist in norm_params, ignored')
                raise KeyError
        return df
    def postprocess(self, df):
        for col in df.columns:
            param_key = str(col)     
            if col[1] in ('year','DOY','hour'):
                pass
            elif param_key in self.norm_params:
                df[col] = self.norm.denormalize(df[col], *self.norm_params[param_key])
                # df[col] = self.norm.normalize(df[col], *self.norm_params)
            else:
                print(f'key {param_key} not exist in norm_params, ignored')
                raise KeyError
        return df
        
        