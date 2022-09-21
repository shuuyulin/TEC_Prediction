from .LSTMTEC import LSTMTEC
from .LSTM_seq2seq import LSTM_Seq2Seq
from .Transformer_encoder import Transformer_encoder
from .Transformer import Transformer
import torch
from pathlib import Path

def initialize_model(config, arg, *args, **kwargs):
    model_list = {
        'LSTM_TEC':LSTMTEC,
        'LSTM_TEC_2SW':LSTMTEC,
        'LSTM_Seq2Seq_TEC': LSTM_Seq2Seq,
        'LSTM_Seq2Seq_TEC_2SW' : LSTM_Seq2Seq,
        'LSTM_Seq2Seq_TEC_5SW' : LSTM_Seq2Seq,
        'Transformer_encoder_GTEC' : Transformer_encoder,
        'Transformer_GTEC' : Transformer,
    }
    model_ft_list = {
        'LSTM_TEC' : 1,
        'LSTM_TEC_2SW' : 3,
        'LSTM_Seq2Seq_TEC' : 1,
        'LSTM_Seq2Seq_TEC_2SW' : 3,
        'LSTM_Seq2Seq_TEC_5SW' : 6,
        'Transformer_encoder_GTEC' : 71*72,
        'Transformer_GTEC' : 71*72,
    }
    
    model_name = config['model']['model_name']
    feature_dim = model_ft_list[model_name]
    
    if arg.mode == 'train':
        model = model_list[model_name](config, feature_dim, *args, **kwargs)
        if arg.checkpoint is not None:
            print(f'Using checkpoint {arg.checkpoint}')
            model.load_state_dict(torch.load(arg.checkpoint))
        return model
    else: # test
        model = model_list[model_name](config, feature_dim, *args, **kwargs)
        model.load_state_dict(torch.load(Path(arg.record) / 'best_model.pth'))
        return model
    