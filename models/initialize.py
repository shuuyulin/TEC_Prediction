from .LSTM import single_point_LSTM
import torch
def initialize_model(config, arg, *args, **kwargs):
    model_list = {
        'single_point_LSTM':single_point_LSTM,
    }
    if arg.mode == 'train':
        return single_point_LSTM(config, *args, **kwargs)
    else: # test
        model = single_point_LSTM(config, *args, **kwargs)
        model.load_state_dict(torch.load(arg.model_path))
        return model
    