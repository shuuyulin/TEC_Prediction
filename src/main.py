import argparse
import configparser
from pathlib import Path
import wandb
import logging
import torch
from .utils import *
from tqdm.auto import tqdm
from .initialization import initialize_all
from .output import exporting
import math

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.ini')
    parser.add_argument('-m', '--mode', type=str, default='train')
    parser.add_argument('-tf', '--truth_path', type=str, default='./data/SWGIM3.0_year')
    parser.add_argument('-r','--record', type=str, default='./') # record path
    parser.add_argument('-ck','--checkpoint', type=str) # continue training, ignore in testing
    parser.add_argument('-i','--run_id', type=str) # continue training, ignore in testing
    parser.add_argument('-o','--output', type=str, default="prediction_frame1.csv")
    
    return parser

def get_config(args):
    
    # configparser
    config = configparser.ConfigParser()
    config.read(args.config)
    
    # device
    if not torch.cuda.is_available():
        config['global']['device'] = 'cpu'
                   
    return config

def main():
    # argparser
    args = get_parser().parse_args()
    config = get_config(args)
    
    # paths
    RECORDPATH = get_record_path(args)
    OUTPUTPATH = RECORDPATH / args.output
    logging.basicConfig(level=logging.INFO)
    
    setSeed(config.getint('global', 'seed'))
    
    if args.mode == 'train':
        wandb.init(
            name=f'{args.record.split("/")[-1]}',
            project='GTEC_prediction',
            config=config,
            id=args.run_id,
            resume='must' if args.run_id else None,
            
        )
    essense = initialize_all(args, config)
            
    if args.mode == 'train':
        training(config, essense, RECORDPATH)
    elif args.mode == 'test':
        evaling(config, essense, OUTPUTPATH)
        
def training(config, essense, RECORDPATH):
    # wandb.watch(essense['model'], essense['criterion'], log='gradients', log_freq=100)
    bestloss, bepoch = 100, 0
    # Running epoch
    for epoch in range(config.getint('train', 'epoch')):
        print(f'Epoch: {epoch}')
        tr_loss = train_one(config, essense)
        vl_loss, _ = eval_one(config, essense)
        if config['train']['lr_scheduler'] != 'OneCycleLR':essense['scheduler'].step(vl_loss)
        if vl_loss <= bestloss:
            bestloss = vl_loss
            bepoch = epoch
            torch.save({'model':essense['model'].state_dict(),
                        'optimizer':essense['optimizer'].state_dict()},
                       RECORDPATH / Path('best_model_ck.pth'))
        # if (epoch+1) % 25 == 0:
        #     torch.save({'model':essense['model'].state_dict()}, RECORDPATH / Path(f'model_ck_{epoch}.pth'))
            
        wandb.log({'lr':math.log10(get_lr(essense['optimizer']))})
    print(f'best valid loss: {bestloss}, epoch: {bepoch}')

def evaling(config, essense, OUTPUTPATH):
    loss, pred = eval_one(config, essense, 'test')
    
    logging.info(f'test unpostprocessed loss: {loss}')
    # np.save(open(RECORDPATH / 'predict.npy', 'wb'), pred)
    # pred = np.load(open(RECORDPATH / 'predict.npy', 'rb'))
    # print(np.shape(pred))
    # print(pred[0])
    exporting(config, pred, essense['df'], OUTPUTPATH)

def train_one(config, essense):
    essense['model'].train()
    totalloss = 0
    with tqdm(essense['train_loader'], unit='batch', desc='Train',dynamic_ncols=True) as tqdm_loader:
        for idx, data in enumerate(tqdm_loader):
            for d in data:
                data[d] = data[d].to(device=config['global']['device'])
            
            output, loss_dict = essense['model'](**data)
            
            essense['optimizer'].zero_grad()
            loss_dict['loss'].backward()
                        
            essense['optimizer'].step()
            if config['train']['lr_scheduler'] == 'OneCycleLR':
                essense['scheduler'].step()

            nowloss = loss_dict['loss'].item()

            totalloss += nowloss
            tqdm_loader.set_postfix(loss=f'{nowloss:.7f}', avgloss=f'{totalloss/(idx+1):7f}')
            
            essense['model'].record(loss_dict, 'train')
            
    return totalloss/len(tqdm_loader)

def eval_one(config, essense, mode='valid'):
    essense['model'].eval()
    output_list = []
    totalloss = 0
    with torch.no_grad():
        with tqdm(essense['eval_loader'],unit='batch',desc=mode, dynamic_ncols=True) as tqdm_loader:
            for idx, data in enumerate(tqdm_loader):
                for d in data:
                    data[d] = data[d].to(device=config['global']['device'])
                
                output, loss_dict = essense['model'](**data)
                #output shape(batch, output_step or 1 , 71*72)
                # print(output.shape)
                output_list.append(output.detach().cpu().numpy())
                
                nowloss = loss_dict['loss'].item()
                totalloss += nowloss
                tqdm_loader.set_postfix(loss=f'{nowloss:.7f}', avgloss=f'{totalloss/(idx+1):.7f}')
                
                if mode == 'valid':
                    essense['model'].record(loss_dict, mode)
    tec_pred_list = np.concatenate(output_list, axis=0) # (len, 71*72) or (len, output_step, 71*72)
        
    return totalloss/len(tqdm_loader), tec_pred_list

if __name__ == '__main__':
    main()
