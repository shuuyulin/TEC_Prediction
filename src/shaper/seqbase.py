import torch
from src.utils import *
class shaper(): # for tec value
    def __init__(self, config) -> None:
        self.config = config
        self.in_ft_len = len([1 for k in config2strlist(self.config['data']['input_features']) if k not in ['tec','tec_er','tec_sh','tec_sh_er']])
        self.out_ft_len = len([1 for k in config2strlist(self.config['data']['truth_features']) if k not in ['tec','tec_er','tec_sh','tec_sh_er']])
        self.i_step = int(config['model']['input_time_step'])
        
    def shape_input(self, tec:torch.Tensor ):
        return tec
    
    def model_tec_drop(self, out):
        return out
    def model_other_drop(self, out):
        return out[:,-1]
    
    def get_input_dim(self):
        return 1
    
    def get_tec_output_dim(self):
        return 71*72
    
    def get_other_output_dim(self):
        return self.out_ft_len
    
    def get_BOS(self, x):
        return x #TODO

class time_shaper(shaper):
    def __init__(self, config) -> None:
        super().__init__(config)
    
    def shape_input(self, tec:torch.Tensor ):
        return tec
    
    def model_tec_drop(self, out):
        # out: batch, i_step, 71 * 72
        # return: batch, 71 * 72
        return out[:,-1]
    
    def model_other_drop(self, out):
        # out: batch, i_step, n_ft_out
        # return: batch, n_ft_out
        return out[:,-1]
    
    def get_input_dim(self):
        return (71*72 + self.in_ft_len) * self.i_step
    
    def get_tec_output_dim(self):
        return 71*72
    
    def get_BOS(self, x):
        return x

class lat_shaper(shaper):
    def __init__(self, config) -> None:
        super().__init__(config)
    
    def shape_input(self, tec:torch.Tensor ):
        return torch.permute(tec.view(-1, 71, 72), (1, 0, 2)).reshape(71, -1)
    
    def model_tec_drop(self, out):
        # out: batch, 71, 72
        # return: batch, 71*72
        return out.view(-1, 71*72)
    def model_other_drop(self, out):
        # out: batch, 71, n_ft_out
        # return: batch, n_ft_out
        return torch.mean(out, dim=1)
    
    def get_input_dim(self):
        return (72 + self.in_ft_len) * self.i_step
    def get_tec_output_dim(self):
        return 72
    
    def get_BOS(self, x):
        return x

class long_shaper(shaper):
    def __init__(self, config) -> None:
        super().__init__(config)
    
    def shape_input(self, tec:torch.Tensor ):
        return torch.permute(tec.view(-1, 71, 72), (2, 0, 1)).reshape(72, -1)
    
    def model_tec_drop(self, out):
        # out: batch, 72, 71
        # return: batch, 71*72
        return torch.permute(out, (0, 2, 1)).reshape(-1, 71*72)
    
    def model_other_drop(self, out):
        # out: batch, 72, n_ft_out
        # return: batch, n_ft_out
        return torch.mean(out, dim=1)
    
    def get_input_dim(self):
        return (71 + self.in_ft_len) * self.i_step
    
    def get_tec_output_dim(self):
        return 71
    
    def get_BOS(self, x):
        return x