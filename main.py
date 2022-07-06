import argparse
import configparser
from pathlib import Path
from utils import *
from preprocessing import initialize_normalization
from dataset import initialize_dataset
from training_tools import initialize_criterion, initialize_optimizer, initialize_lr_scheduler
from models import initialize_model

# Paths
BASEPATH = Path(__file__).parent
DATAPATH = BASEPATH / Path('raw_data/SWGIM_year')
CONFIGPATH = BASEPATH / Path('config.ini')
RECORDPATH = BASEPATH / Path('./record')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--model_path', type=str)
    
    return parser

def main():
    # argparser
    parser = get_parser()
    args = parser.parse_args()
    
    # configparser
    config = configparser.ConfigParser()
    config.read(CONFIGPATH)
    
    # record path
    global RECORDPATH
    RECORDPATH = get_record_path(RECORDPATH, args)
    
    # device
    if not torch.cuda.is_available():
        config['global']['device'] = 'cpu'
        
    rd_seed = config.getint('global', 'seed')
    setSeed(rd_seed)
    
    # read csv data
    df = read_csv_data(config, args.mode, DATAPATH)
    
    # print(df.head())
    # print(df.info())
    # preprocessing
    norm = initialize_normalization(config)
    
    df[df.columns[3: len(df.columns)]] = norm.fit_transform(df[df.columns[3: len(df.columns)]])
        
    # get dataloader
    if args.mode == 'train':
    
        train_indices, valid_indices = get_indices(config, df, rd_seed, 'train')
        train_loader = initialize_dataset(config, df, train_indices, task="train")
        valid_loader = initialize_dataset(config, df, valid_indices, task="eval")
    elif args.mode == 'test':
        
        test_indices = get_indices(config, df, rd_seed, 'test')
        test_loader = initialize_dataset(config, df, test_indices, task="eval")
    else:
        raise AttributeError(f'no mode name: {args.mode}')
    
    print(f'indices len: {len(test_indices)}')
    
    # for idx, data in enumerate(test_loader):
    #     if idx <= 0:
    #         print(f'{idx} {data}')
    
    criterion = initialize_criterion(config)
    model = initialize_model(config, args, criterion).to(device=config['global']['device'])
    
    if args.mode == 'train':
        optimizer = initialize_optimizer(config, model.parameters())
        scheduler = initialize_lr_scheduler(config, optimizer)
        
        # ==TODO== tensorboard recording
        tr_losses, vl_losses, lrs = [], [], []
        bestloss, bepoch = 100, 0
        # Running epoch
        for epoch in range(config.getint('train', 'epoch')):
            print(f'epoch: {epoch}')
            tr_loss = train_one(config, norm, model, train_loader, optimizer, scheduler)
            vl_loss = eval_one(config, norm, model, valid_loader)
            lrs.append(get_lr(optimizer))
            scheduler.step(vl_loss)
            tr_losses.append(tr_loss)
            vl_losses.append(vl_loss)
            if vl_loss <= bestloss:
                bestloss = vl_loss
                bepoch = epoch
                torch.save(model.state_dict(), RECORDPATH / Path('best_model.pth'))
                
        print(f'best valid loss: {bestloss}, epoch: {bepoch}')

        # Plot train/valid loss, accuracy, learning rate
        plot_fg(tr_losses, 'losses', 'loss', RECORDPATH, vl_losses)
        plot_fg(lrs, 'lrs', 'lr', RECORDPATH)
        
    elif args.mode == 'test':
        loss, pred = eval_one(config, norm, model, test_loader, 'Test')
        print(f'test loss: {loss}')
        print(pred.shape)
        pd.DataFrame(pred, columns=['predict']).to_csv(RECORDPATH / Path('predition.csv'))
        
def train_one(config, norm, model, dataloader, optimizer, scheduler=None):
    model.train()
    totalloss = 0
    with tqdm(dataloader, unit='batch', desc='Train') as tqdm_loader:
        for idx, data in enumerate(tqdm_loader): # ==TODO== space_data ignored
            for d in data:
                data[d] = data[d].to(device=config['global']['device'])
                
            output, loss = model(**data)
            
            optimizer.zero_grad()
            loss.backward()
                        
            optimizer.step()
            # scheduler.step()

            nowloss = loss.item()

            totalloss += nowloss
            tqdm_loader.set_postfix(loss=f'{nowloss:.3f}', avgloss=f'{totalloss/(idx+1):3f}')
    return totalloss/len(tqdm_loader)

def eval_one(config, norm, model, dataloader, mode='Valid'):
    model.eval()
    output_list = []
    totalloss, bestloss = 0, 10
    with torch.no_grad():
        with tqdm(dataloader,unit='batch',desc=mode) as tqdm_loader:
            for idx, data in enumerate(tqdm_loader):
                for d in data:
                    data[d] = data[d].to(device=config['global']['device'])

                output, loss = model(**data) #output shape(4, 1)
                # print(output.shape) 
                output_list.append(output.detach().cpu().view(-1).numpy())
                
                nowloss = loss.item()
                totalloss += nowloss
                tqdm_loader.set_postfix(loss=f'{nowloss:.3f}', avgloss=f'{totalloss/(idx+1):3f}')
    tec_pred_list = np.concatenate(output_list, axis=0)
    
    # denormalize
    tec_pred_list = norm.inverse_transform(tec_pred_list.reshape((-1,1)))
    print(tec_pred_list.shape)
    
    return totalloss/len(tqdm_loader), tec_pred_list

if __name__ == '__main__':
    main()
