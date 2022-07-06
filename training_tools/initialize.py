import torch.nn as nn
import torch.optim as optim
def initialize_criterion(config, *args, **kwargs):
    criterion_type_list = {
        'MSELoss': nn.MSELoss,
        'L1Loss': nn.L1Loss,
    }
    criterion_type = config['train']['criterion']
    
    if criterion_type in criterion_type_list:
        criterion = criterion_type_list[criterion_type](*args, **kwargs)
    else:
        print('criterion_type has not been defined in config file!')
        raise AttributeError

    return criterion

def initialize_optimizer(config, *args, **kwargs):
    optimizer_type_list = {
        'AdamW': optim.AdamW,
        'Ranger': None, # ==TODO==
    }
    optimizer_type = config['train']['optimizer']
    
    if optimizer_type not in optimizer_type_list:
        print('optimizer_type has not been defined in config file!')
        raise AttributeError
    elif optimizer_type == 'AdamW':
        optimizer = optimizer_type_list[optimizer_type](eps=1e-8, *args, **kwargs)
    # elif

    return optimizer

def initialize_lr_scheduler(config, *args, **kwargs):
    lr_scheduler_type_list = {
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau,
        'CosineAnnealingLR':optim.lr_scheduler.CosineAnnealingLR,
    }
    lr_scheduler_type = config['train']['lr_scheduler']
    
    if lr_scheduler_type not in lr_scheduler_type_list:
        print('lr_scheduler_type has not been defined in config file!')
        raise AttributeError
    elif lr_scheduler_type == 'ReduceLROnPlateau':
        lr_scheduler = lr_scheduler_type_list[lr_scheduler_type](mode='min', patience=2,
                        factor=0.1, min_lr=config.getfloat('train', 'lr')*1e-5, *args, **kwargs)
    elif lr_scheduler_type == 'CosineAnnealingLR':
        lr_scheduler = lr_scheduler_type_list[lr_scheduler_type](T_max= 2, eta_min=1e-65, *args, **kwargs)
        
    return lr_scheduler

