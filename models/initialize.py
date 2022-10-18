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
        'Transformer_encoder_GTEC' : Transformer_encoder,
        'Transformer_GTEC' : Transformer,
    }
    
    # model_ft_list = {
    #     'LSTM_TEC' : 1,
    #     'LSTM_TEC_2SW' : 3,
    #     'LSTM_Seq2Seq_TEC' : 1,
    #     'LSTM_Seq2Seq_TEC_2SW' : 3,
    #     'LSTM_Seq2Seq_TEC_5SW' : 6,
    #     'Transformer_encoder_GTEC' : 71*72,
    #     'Transformer_GTEC' : 71*72,
    # }
    
    model_name = config['model']['model_name']
    seq_base = config['data']['seq_base']
    features = config2strlist(config['data']['features'])
    input_time_step = int(config['model']['input_time_step'])
    
    # input_dim = model_ft_list[model_name]
    if seq_base == 'time':
        input_dim = len(features) - 1 + (71*72 if features[-1] == 'tec' else 256)
        output_dim = 71*72
    elif seq_base == 'latitude':
        input_dim = len(features) - 1 + 72 * input_time_step
        output_dim = 72
    elif seq_base == 'longtitude':
        input_dim = len(features) - 1 + 71 * input_time_step
        output_dim = 71
    else:
        print('seq_base has not been defined in config file!')
        raise AttributeError
    
    if arg.mode == 'train':
        model = model_list[model_name](config, arg, input_dim, output_dim, *args, **kwargs)
        if arg.checkpoint is not None:
            print(f'Using checkpoint {arg.checkpoint}')
            model.load_state_dict(torch.load(arg.checkpoint))
        return model
    else: # test
        model = model_list[model_name](config, arg, input_dim, output_dim, *args, **kwargs)
        model.load_state_dict(torch.load(Path(arg.record) / 'best_model.pth'))
        return model
    