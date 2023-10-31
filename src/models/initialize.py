from .LSTMTEC import LSTMTEC
from .LSTM_seq2seq import LSTM_Seq2Seq
from .Transformer_encoder import Transformer_encoder
from .Transformer import Transformer
from .Transformer_E_multitask import Transformer_E_mttasks
import torch
from pathlib import Path
from ..utils import *
import logging

def initialize_model(config, arg, *args, **kwargs):
    model_list = {
        'LSTM_TEC':LSTMTEC,
        'LSTM_Seq2Seq_TEC': LSTM_Seq2Seq,
        'Transformer_E' : Transformer_encoder,
        'Transformer_ED' : Transformer,
        'Transformer_E_mttasks' : Transformer_E_mttasks,
    }
        
    model_name = config['model']['model_name']
    # seq_base = config['data']['seq_base']
    input_features = config2strlist(config['data']['input_features'])
    # global_features = config2strlist(config.get('data', 'global_features'))
    # tec_features = config['data']['tec_features']
    # seq_pos_feature = config.getboolean('data', 'seq_feature')
    
    input_time_step = int(config['model']['input_time_step'])
    # if seq_base == 'time':
    #     input_dim = 2*len(date_features) + len(global_features) + (71*72 if tec_features == 'tec' else 256)
    #     output_dim = 71*72
    # elif seq_base == 'latitude':
    #     input_dim = 2*len(date_features) + (len(global_features) + 72) * input_time_step
    #     output_dim = 72
    # elif seq_base == 'longitude':
    #     # FLAG: feature normalization min-max / sin-cos
    #     input_dim = len(date_features) + (len(global_features) + 71) * input_time_step
    #     output_dim = 71
    # else:
    #     logging.error('seq_base has not been defined in config file!')
    #     raise AttributeError
    
    if arg.mode == 'train':
        model = model_list[model_name](config, *args, **kwargs)
        if arg.checkpoint is not None:
            logging.info(f'Using model checkpoint {arg.checkpoint}')
            model.load_state_dict(torch.load(arg.checkpoint)['model'])
            model = model.to(device=config['global']['device'])
        return model
    else: # test
        model = model_list[model_name](config, *args, **kwargs)
        
        path = arg.checkpoint if arg.checkpoint else Path(arg.record) / 'best_model_ck.pth'
        model.load_state_dict(torch.load(path)['model'])
        model = model.to(device=config['global']['device'])
        return model
    