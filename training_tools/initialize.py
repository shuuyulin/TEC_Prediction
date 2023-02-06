import torch
import torch.nn as nn
import torch.optim as optim
from .criterion import RMSELoss

def initialize_criterion(config, *args, **kwargs):
    criterion_type_list = {
        'MSELoss': nn.MSELoss,
        'RMSELoss': RMSELoss,
        'L1Loss': nn.L1Loss,
        'SmoothL1Loss': nn.SmoothL1Loss,
    }
    criterion_type = config['train']['criterion']
    
    if criterion_type in criterion_type_list:
        criterion = criterion_type_list[criterion_type](*args, **kwargs)
    else:
        print('criterion_type has not been defined in config file!')
        raise AttributeError

    return criterion

def initialize_optimizer(config, arg, *args, **kwargs):
    optimizer_type_list = {
        'SGD': optim.SGD,
        'AdamW': optim.AdamW,
        'Ranger': None, # ==TODO==
    }
    optimizer_type = config['train']['optimizer']
    
    if optimizer_type not in optimizer_type_list:
        print('optimizer_type has not been defined in config file!')
        raise AttributeError
    elif optimizer_type == 'AdamW':
        optimizer = optimizer_type_list[optimizer_type](eps=1e-8, lr=float(config['train']['lr']), *args, **kwargs)
    elif optimizer_type == 'SGD':
        optimizer = optimizer_type_list[optimizer_type](momentum=0.9, lr=float(config['train']['lr']), *args, **kwargs)
    
    if arg.optimizer_checkpoint is not None:
            print(f'Using optimizer checkpoint {arg.optimizer_checkpoint}')
            optimizer.load_state_dict(torch.load(arg.optimizer_checkpoint))
    return optimizer

def initialize_lr_scheduler(config, steps_pep, *args, **kwargs):
    lr_scheduler_type_list = {
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau,
        'CosineAnnealingLR':optim.lr_scheduler.CosineAnnealingLR,
        'OneCycleLR':optim.lr_scheduler.OneCycleLR,
    }
    lr_scheduler_type = config['train']['lr_scheduler']
    
    if lr_scheduler_type not in lr_scheduler_type_list:
        print('lr_scheduler_type has not been defined in config file!')
        raise AttributeError
    elif lr_scheduler_type == 'ReduceLROnPlateau':
        lr_scheduler = lr_scheduler_type_list[lr_scheduler_type](mode='min', patience=4,# threshold=1e-3,
                        factor=0.1, min_lr=config.getfloat('train', 'lr')*1e-10, *args, **kwargs)
    elif lr_scheduler_type == 'CosineAnnealingLR':
        lr_scheduler = lr_scheduler_type_list[lr_scheduler_type](T_max= 2, eta_min=1e-65, *args, **kwargs)
    elif lr_scheduler_type == 'OneCycleLR':
        ep = int(config['train']['epoch'])
        lr_scheduler = lr_scheduler_type_list[lr_scheduler_type](max_lr=0.01, epochs=ep,\
                                                    steps_per_epoch=steps_pep, *args, **kwargs)
        
    return lr_scheduler

