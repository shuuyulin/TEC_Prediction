# Formatter is a funtion for dataloader.
# It drops and renames dataframe columns.
#
from utils import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def single_point_LSTM_formatter(batch): # ingore space weather
    
    _, tec, truth = zip(*batch)
    
    tec_lens = [len(t) for t in tec]
    
    tec_pad = pad_sequence(tec, batch_first=True, padding_value=-1)
    truth_pad = pad_sequence(truth, batch_first=True, padding_value=-1)
    
    tec_packed = pack_padded_sequence(tec_pad, tec_lens, batch_first=True, enforce_sorted=False)

    return {
        'tec_packed':tec_packed,
        'truth_pad':truth_pad,
    }

# class BasicFormatter:
#     def __init__(self, config, *args, **params):
#         self.config = config

#     def process(self, data, *args, **params):
#         return data
# class single_point_LSTM_formatter(BasicFormatter):
#     def __init__(self, config, *args, **params):
#         self.config = config
        
#     def process(self, data, *args, **params):
        
#         return data

# # ==TODO== global
# class global_formatter(BasicFormatter):
#     def __init__(self, config, *args, **params):
#         self.config = config

#     def process(self, data, *args, **params):
        
#         return data