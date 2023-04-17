from .LSTMTEC import LSTMTEC
from .LSTM_seq2seq import LSTM_Seq2Seq
from .Transformer_encoder import Transformer_encoder
from .Transformer import Transformer
import torch
from pathlib import Path
from utils import *

def initialize_model(config, arg, *args, **kwargs):
    model_list = {
        'LSTM_TEC':LSTMTEC,
        'LSTM_Seq2Seq_TEC': LSTM_Seq2Seq,
        'Transformer_E' : Transformer_encoder,
        'Transformer_ED' : Transformer,
    }
        
    model_name = config['model']['model_name']
    seq_base = config['data']['seq_base']
    date_features = config2strlist(config['data']['date_features'])
    global_features = config2strlist(config.get('data', 'global_features'))
    tec_features = config['data']['tec_features']
    seq_pos_feature = config.getboolean('data', 'seq_feature')
    
    input_time_step = int(config['model']['input_time_step'])
    s_len = 2 if config['data']['date_seq_base_norm'] == "Hibert" else 1

    if seq_base == 'time':
        input_dim = s_len*len(date_features) + len(global_features) + (71*72 if tec_features == 'tec' else 256) + s_len*seq_pos_feature
        output_dim = 71*72
    elif seq_base == 'latitude':
        input_dim = s_len*len(date_features) + (len(global_features) + 72) * input_time_step + s_len*seq_pos_feature
        output_dim = 72
    elif seq_base == 'longitude':
        # FLAG: feature normalization min-max / sin-cos
        input_dim = s_len*len(date_features) + (len(global_features) + 71) * input_time_step + s_len*seq_pos_feature
        output_dim = 71
    else:
        print('seq_base has not been defined in config file!')
        raise AttributeError
    
    if arg.mode == 'train':
        model = model_list[model_name](config, arg, input_dim, output_dim, *args, **kwargs)
        if arg.model_checkpoint is not None:
            print(f'Using model checkpoint {arg.model_checkpoint}')
            model.load_state_dict(torch.load(arg.model_checkpoint))
        return model
    else: # test
        model = model_list[model_name](config, arg, input_dim, output_dim, *args, **kwargs)
        model.load_state_dict(torch.load(Path(arg.record) / 'best_model.pth',
                                         map_location=torch.device(config['global']['device'])))
        return model
    