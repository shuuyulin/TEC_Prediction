import argparse
import configparser
from pathlib import Path
from utils import *
from tqdm.auto import tqdm
from preprocessing import initialize_processer
from dataset import initialize_dataset
from training_tools import initialize_criterion, initialize_optimizer, initialize_lr_scheduler
from models import initialize_model
from output import exporting
import copy

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.ini')
    parser.add_argument('-m', '--mode', type=str, default='train')
    parser.add_argument('-r','--record', type=str) # record path
    parser.add_argument('-k','--model_checkpoint', type=str) # continue training, ignore in testing
    parser.add_argument('-o','--optimizer_checkpoint', type=str) # continue training, ignore in testing
    parser.add_argument('-t','--retest_train_data', action='store_true')
    parser.add_argument('-s','--shifting_cnt', type=int, default=1)
    return parser

def get_config(args):
    
    # configparser
    config = configparser.ConfigParser()
    config.read(args.config)
    
    # device
    if not torch.cuda.is_available():
        config['global']['device'] = 'cpu'
        
    if config['preprocess']['normalization_type'] == 'None':
        config['preprocess']['predict_norm'] = 'False'
       
    if args.retest_train_data:
        config['data']['test_year'] = "2018, 2019"
    
    if args.shifting_cnt != 1:
        if args.mode != 'test':
            print('Mode error, shifting test must be in test mode')
            raise AttributeError

    return config

def main():
    # argparser
    args = get_parser().parse_args()
    config = get_config(args)
    
    # paths
    RECORDPATH = get_record_path(args)
    DATAPATH = './data/'
            
    rd_seed = config.getint('global', 'seed')
    setSeed(rd_seed)
    
    # read csv data
    df, truth_df = read_csv_data(config, args.mode, DATAPATH)
    pred_df = copy.deepcopy(df)
    
    # preprocessing
    processer = initialize_processer(config)
    
    if config['preprocess']['normalization_type'] != 'None':
        df = processer.preprocess(df)
                    
    if config['preprocess']['predict_norm'] == 'True':
        truth_df = processer.preprocess(truth_df)
        
    if args.mode == 'train':
    
        train_indices, valid_indices = get_indices(config, df, rd_seed, 'train')
        train_loader = initialize_dataset(config, df, truth_df, train_indices, processer, task="train")
        valid_loader = initialize_dataset(config, df, truth_df, valid_indices, processer, task="eval")
        print(f'indices len: {len(train_indices)}, {len(valid_indices)}')
    elif args.mode == 'test':
        
        test_indices = get_indices(config, df, rd_seed, 'test')
        test_loader = initialize_dataset(config, df, truth_df, test_indices, processer, task="eval")
        print(f'indices len: {len(test_indices)}')
    else:
        raise AttributeError(f'no mode name: {args.mode}')
    
    
    criterion = initialize_criterion(config)
    model = initialize_model(config, args, criterion).to(device=config['global']['device'])
    
    if args.mode == 'train':
        optimizer = initialize_optimizer(config, args, model.parameters())
        scheduler = initialize_lr_scheduler(config, len(train_loader),  optimizer)
        
        # TODO: tensorboard recording
        tr_losses, vl_losses, lrs = [], [], []
        bestloss, bepoch = 100, 0
        # Running epoch
        for epoch in range(config.getint('train', 'epoch')):
            print(f'epoch: {epoch}')
            tr_loss = train_one(config, model, train_loader, optimizer, scheduler)
            vl_loss, _ = eval_one(config, model, valid_loader)
            lrs.append(get_lr(optimizer))
            if config['train']['lr_scheduler'] != 'OneCycleLR':scheduler.step(vl_loss)
            tr_losses.append(tr_loss)
            vl_losses.append(vl_loss)
            if vl_loss <= bestloss:
                bestloss = vl_loss
                bepoch = epoch
                torch.save(model.state_dict(), RECORDPATH / Path('best_model.pth'))
                torch.save(optimizer.state_dict(), RECORDPATH / Path('optimizer.pth'))

        print(f'best valid loss: {bestloss}, epoch: {bepoch}')
 
        # Plot train/valid loss, accuracy, learning rate
        plot_fg(tr_losses, 'losses', 'loss', RECORDPATH, vl_losses)
        plot_fg(np.log10(lrs), 'lrs', 'log(lr)', RECORDPATH)
        
    elif args.mode == 'test':
        if args.shifting_cnt > 1:
            loss, pred = shifting_test_one(args, config, model, test_loader, processer, RECORDPATH, 'Test')
        else:    
            loss, pred = eval_one(config, model, test_loader, 'Test')
        
        print(f'test unpostprocessed loss: {loss}')
        # np.save(open(RECORDPATH / 'predict.npy', 'wb'), pred)
        # pred = np.load(open(RECORDPATH / 'predict.npy', 'rb'))
        exporting(args, config, pred, pred_df, processer, RECORDPATH)

def train_one(config, model, dataloader, optimizer, scheduler=None):
    model.train()
    totalloss = 0
    with tqdm(dataloader, unit='batch', desc='Train',dynamic_ncols=True) as tqdm_loader:
        for idx, data in enumerate(tqdm_loader):
            for d in data:
                data[d] = data[d].to(device=config['global']['device'])
                
            output, loss = model(**data)
            
            optimizer.zero_grad()
            loss.backward()
                        
            optimizer.step()
            if config['train']['lr_scheduler'] == 'OneCycleLR':
                scheduler.step()

            nowloss = loss.item()

            totalloss += nowloss
            tqdm_loader.set_postfix(loss=f'{nowloss:.7f}', avgloss=f'{totalloss/(idx+1):7f}')
    return totalloss/len(tqdm_loader)

def eval_one(config, model, dataloader, mode='Valid'):
    model.eval()
    output_list = []
    totalloss, bestloss = 0, 10
    with torch.no_grad():
        with tqdm(dataloader,unit='batch',desc=mode, dynamic_ncols=True) as tqdm_loader:
            for idx, data in enumerate(tqdm_loader):
                for d in data:
                    data[d] = data[d].to(device=config['global']['device'])
                output, loss = model(**data)
                #output shape(batch, output_step or 1 , 71*72)
                
                output_list.append(output.detach().cpu().numpy())
                
                nowloss = loss.item()
                totalloss += nowloss
                tqdm_loader.set_postfix(loss=f'{nowloss:.7f}', avgloss=f'{totalloss/(idx+1):.7f}')
    tec_pred_list = np.concatenate(output_list, axis=0) # (len, 71*72) or (len, output_step, 71*72)
        
    return totalloss/len(tqdm_loader), tec_pred_list

def shifting_test_one(args, config, model, dataloader, processer, RECORDPATH, mode='Test'): # only for longitude
    model.eval()
    output_list = []
    totalloss, bestloss = 0, 10
    with torch.no_grad():
        with tqdm(dataloader,unit='batch',desc=mode, dynamic_ncols=True) as tqdm_loader:
            for idx, data in enumerate(tqdm_loader):
                for d in data:
                    data[d] = data[d].to(device=config['global']['device'])
                
                output_timestep = np.empty((config.getint('eval', 'batch_size'), args.shifting_cnt, 71*72), dtype=np.float32)
                for i in range(args.shifting_cnt):
                    output, loss = model(**data)
                    output_timestep[:,i] = output.detach().cpu().numpy()
                    torch.set_printoptions(threshold=10_000)
                    # print('output', output.shape)
                    if config['preprocess']['normalization_type'] != 'None' and config['preprocess']['predict_norm'] == 'False':
                        output = processer.preprocess_t(output).to(config['global']['device'])
                    output = torch.permute(output.view(8, -1, 71, 72), (0, 3, 1, 2)).reshape(8, 72, -1)
                    
                    data['x'] = torch.cat((data['x'][:,:,71:], output), dim=2)
                    del output
                        
                #output shape(batch, shifting_cnt, 71*72)
                
                output_list.append(output_timestep) # (365+366)*24, 96, 71*72
                
                nowloss = loss.item()
                totalloss += nowloss
                tqdm_loader.set_postfix(loss=f'{nowloss:.7f}', avgloss=f'{totalloss/(idx+1):.7f}')
            
                if idx % 100 == 99 or idx + 1 == len(tqdm_loader):
                    tec_pred_list = np.concatenate(output_list, axis=0) #(len, shifting_cnt, 71*72)
                    np.save(open(RECORDPATH / f'predict{idx}.npy', 'wb'), tec_pred_list)
                    # print("filenam: ", RECORDPATH / f'predict{idx}.npy')
                    output_list = []
                    del tec_pred_list
        
    return totalloss/len(tqdm_loader), tec_pred_list
if __name__ == '__main__':
    main()
