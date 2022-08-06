from .dataset import SWGIMDataset
from torch.utils.data import DataLoader
from .collate_fn import TEC_formatter, TEC_2SW_formatter
def initialize_dataset(config, *args, **kwargs):
    dataset_type_list = {
        'SWGIMDataset': SWGIMDataset,
    }
    collate_fn_type_list = {
        'LSTM_TEC' : TEC_formatter,
        'LSTM_TEC_2SW' : TEC_2SW_formatter,
        'LSTM_Seq2Seq_TEC' : TEC_formatter,
    }
    
    dataset_type = config['data']['dataset_type']
    collate_fn_type = config['model']['model_name']
        
    if dataset_type in dataset_type_list:
        dataset = dataset_type_list[dataset_type](config, *args, **kwargs)
    else:
        print('dataset_type has not been defined in config file!')
        raise AttributeError
    
    if collate_fn_type in collate_fn_type_list:
        collate_fn = collate_fn_type_list[collate_fn_type]
    else:
        print('collate_fn_type has not been defined in config file!')
        raise AttributeError
    
    
    # print(len(dataset))
    # print(dataset[0])
    
    task = kwargs['task']
    drop_last = True if task == 'train' else False
    
    return DataLoader(dataset=dataset,
                      batch_size=config.getint(task, 'batch_size'),
                      shuffle=config.getboolean(task, 'shuffle'),
                      num_workers=config.getint(task, 'num_worker'),
                      collate_fn=collate_fn,
                      drop_last=drop_last)
    