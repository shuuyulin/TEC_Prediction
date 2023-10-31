from src.importer import initialize_import
from src.shaper import initialize_shaper
from src.dataset import initialize_dataset
from src.training_tools import initialize_criterion, initialize_optimizer, initialize_lr_scheduler
# from preprocessing import initialize_processer
from src.models import initialize_model
from src.preprocessing import preprocess_data
import logging
import numpy as np
import pickle
        
def initialize_all(args, config):
    
    all_df = initialize_import(config=config, mode=args.mode, path=args.truth_path).import_data()
    # print(all_df.info())
    data = preprocess_data(config, all_df)
    
    essense = {}
    essense['df'] = all_df
    essense['data'] = data
    essense['shaper'] = initialize_shaper(config)
    essense['criterion'] = initialize_criterion(config)
    essense['model'] = initialize_model(config, args, essense['shaper'], essense['criterion']).to(device=config['global']['device'])
        
    if args.mode == 'train':
    
        train_indices, valid_indices = get_indices(config, data, args.mode)   
        # print(len(train_indices), train_indices)     
        # print(len(valid_indices), valid_indices)
        # pickle.dump(train_indices, open('train_idx.pickle', 'wb'))
        # pickle.dump(valid_indices, open('valid_idx.pickle', 'wb'))
        # train_indices
        # exit()
        essense['train_loader'] = initialize_dataset(config, essense, train_indices, task="train")
        essense['eval_loader'] = initialize_dataset(config, essense, valid_indices, task="eval")
        essense['optimizer'] = initialize_optimizer(config, args, essense['model'].parameters())
        essense['scheduler'] = initialize_lr_scheduler(config, len(essense['train_loader']),  essense['optimizer'])
        
        logging.info(f'indices len: {len(train_indices)}, {len(valid_indices)}')
        
    elif args.mode == 'test':
        
        test_indices = get_indices(config, data, args.mode)
        essense['eval_loader'] = initialize_dataset(config, essense, test_indices, task="eval")
        logging.info(f'indices len: {len(test_indices)}')
        
    return essense

def get_indices(config, data, mode='train', p=0.8):
    # seed = config.getint('global', 'seed')
    
    p = config.getfloat('data', 'valid_ratio')
    
    i_step, o_step = int(config['model']['input_time_step']), int(config['model']['output_time_step'])
    
    task = mode if mode == 'train' else 'eval'
    isShuffle = config[task]['shuffle'] == 'True'
    
    indices = np.arange(len(list(data['input'].values())[0]))
    
    k = len(indices) - i_step - o_step + 1
    # k = len(indices) - i_step - (0 if config['model']['model_name'] == 'Transformer_ED' else o_step - 1)
    indices = indices[:k]
    
    # indices = pickle.load(open('./train_idx_o.pickle', 'rb'))
    if isShuffle:
        np.random.shuffle(indices)
    
    if mode == 'train':
        return indices[:int(len(indices)*p)], indices[int(len(indices)*p):] 
    else:
        return indices