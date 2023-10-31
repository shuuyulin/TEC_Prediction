# Formatter is a funtion for dataloader.
# It drops and renames dataframe columns.
# TODO: fix LSTM formatter
from ..utils import *
# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch

def TEC_formatter(batch): # ignore space weather
    # print(batch)
    x, y = zip(*batch)

    # print(x)    
    # print(y)
    # print(len(x),x[0].shape)
    # print(len(y), y[0].shape)
    #only take last tec
    # print(y[0].keys(), y[0]['tec'].shape)
    y = {k:torch.stack([t[k][-1] for t in y]) for k in y[0]}
    # torch.stack([b.values() for b in y])
    # print(y.keys(), y['dst'].shape)
    # exit()
    x = torch.stack(x)
    # y = torch.stack(y)
    
    return {
        'x':x,
        **y,
    }

def Seq2Seq_TEC_formatter(batch): # ignore space weather
    # print(batch)
    x, y = zip(*batch)
    
    x = torch.stack(x)
    y = {k:torch.stack(y) for k in y[0]} #TODO: possible error
    # y = torch.stack(y)
    
    return {
        'x':x,
        'y':y,
    }
