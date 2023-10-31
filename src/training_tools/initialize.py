import torch
import torch.nn as nn
import torch.optim as optim
from .criterion import *
import logging
from ..utils import *
def initialize_criterion(config, *args, **kwargs):
    criterion_type_list = {
        'MSELoss': nn.MSELoss,
        'RMSELoss': RMSELoss,
        'L1Loss': nn.L1Loss,
        'SmoothL1Loss': nn.SmoothL1Loss,
        # 'multitaskMSELoss': multitaskRegresisonLoss,
    }
    criterion_type = config['train']['criterion']
    
    if criterion_type in criterion_type_list:
        criterion = criterion_type_list[criterion_type](*args, **kwargs)
        
        if len(config2strlist(config['data']['truth_features'])) > 1:
            weight = config2floatlist(config['train']['feature_loss_weight'])
            criterion = multitaskRegresisonLoss(criterion, weight)
    else:
        logging.error('criterion_type has not been defined in config file!')
        raise AttributeError

    return criterion

def initialize_optimizer(config, arg, *args, **kwargs):
    optimizer_type_list = {
        'SGD': optim.SGD,
        'AdamW': optim.AdamW,
        'Ranger': None, # ==TODO==
    }
    type_args = {
        'SGD' : {
            'momentum':0.9,
            'lr':float(config['train']['lr']),
            },
        'AdamW' : {
            'eps':1e-8,
            'lr':float(config['train']['lr']),
            'weight_decay':1e-3,
            },
        'Ranger' : {},
    }
    
    optimizer_type = config['train']['optimizer']
    
    if optimizer_type not in optimizer_type_list:
        logging.error('optimizer_type has not been defined in config file!')
        raise AttributeError
    optimizer = optimizer_type_list[optimizer_type](*args, **type_args[optimizer_type], **kwargs)
    
    if arg.mode != 'test' and arg.checkpoint is not None:
        logging.info(f'Loading optimizer checkpoint from {arg.checkpoint}')
        optimizer.load_state_dict(torch.load(arg.checkpoint)['optimizer'])
        
    return optimizer

def initialize_lr_scheduler(config, steps_pep, *args, **kwargs):
    lr_scheduler_type_list = {
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau,
        'CosineAnnealingLR':optim.lr_scheduler.CosineAnnealingLR,
        'OneCycleLR':optim.lr_scheduler.OneCycleLR,
    }
    lr_scheduler_type = config['train']['lr_scheduler']
    
    type_args = {
        'ReduceLROnPlateau' : {
            'mode':'min',
            'patience':5,
            'threshold':1e-4,
            'eps':0,
            'min_lr':config.getfloat('train', 'lr')*1e-15,
            'factor':0.1,
            'verbose':True,
            },
        'CosineAnnealingLR' : {
            'T_max': 2,
            'eta_min':1e-65,
            },
        'OneCycleLR' : {
            'max_lr':0.01,
            'epochs':int(config['train']['epoch']),
            'steps_per_epoch':steps_pep,
            },
    }
    if lr_scheduler_type not in lr_scheduler_type_list:
        logging.error('lr_scheduler_type has not been defined in config file!')
        raise AttributeError
    
    lr_scheduler = lr_scheduler_type_list[lr_scheduler_type]( *args, **type_args[lr_scheduler_type], **kwargs)
        
    return lr_scheduler

