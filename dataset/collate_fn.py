# Formatter is a funtion for dataloader.
# It drops and renames dataframe columns.
#
from utils import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch

def TEC_formatter(batch): # ignore space weather
    
    _, tec, truth = zip(*batch)
    
    #only take last tec
    truth = [t[-1] for t in truth]
    
    tec_lens = [len(t) for t in tec]
    
    tec_pad = pad_sequence(tec, batch_first=True, padding_value=-1)
    truth_pad = pad_sequence(truth, batch_first=True, padding_value=-1)
    
    tec_packed = pack_padded_sequence(tec_pad, tec_lens, batch_first=True, enforce_sorted=False)

    return {
        'feature_packed':tec_packed,
        'truth_pad':truth_pad,
    }

def TEC_2SW_formatter(batch): # with space weather
    
    SW, tec, truth = zip(*batch)
    
    #only take last tec
    truth = [t[-1] for t in truth]
    
    # TEC_2SW model only takes F10.7 & ap index
    SW = [data[:,[4, 3]] for data in SW]
    
    input_feature = [torch.cat((a,b), dim=1) for a, b in zip(SW, tec)]
    
    feature_lens = [len(t) for t in input_feature]
    
    feature_pad = pad_sequence(input_feature, batch_first=True, padding_value=-1)
    truth_pad = pad_sequence(truth, batch_first=True, padding_value=-1)
    
    feature_packed = pack_padded_sequence(feature_pad, feature_lens, batch_first=True, enforce_sorted=False)

    return {
        'feature_packed':feature_packed,
        'truth_pad':truth_pad,
    }
    
def Seq2Seq_TEC_formatter(batch): # ignore space weather
    # print(batch)
    _, tec, truth = zip(*batch)
    
    tec = torch.stack(tec)
    truth = torch.stack(truth)
    
    return {
        'x':tec,
        'y':truth,
    }

def Seq2Seq_TEC_2SW_formatter(batch): # ignore space weather
    # print(batch)
    sw, tec, truth = zip(*batch)
    
    sw = [data[:,[4, 3]] for data in sw]
    x = torch.stack([torch.cat((a,b), dim=1) for a, b in zip(sw, tec)])
    y = torch.stack(truth)
    
    return {        
        'x':x,
        'y':y,
    }
    
def Seq2Seq_TEC_5SW_formatter(batch): # ignore space weather
    # print(batch)
    sw, tec, truth = zip(*batch)
    
    x = torch.stack([torch.cat((a,b), dim=1) for a, b in zip(sw, tec)])
    y = torch.stack(truth)
    
    return {        
        'x':x,
        'y':y,
    }
    
    
def GTEC_formatter(batch):
    # print(batch)
    _, tec, truth = zip(*batch)
    
    #only take last tec
    truth = [t[-1] for t in truth]
    
    x = torch.stack(tec)
    y = torch.stack(truth)
    
    return {
        'x':x,
        'y':y,
    }