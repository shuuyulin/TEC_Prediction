from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def SWGIM_export(config, rounding_digit, pred, pred_df, processer, RECORDPATH):
    
    SWGIM_model_name = config['model']['model_name'].split('_')[0]
    tmp = int(config['model']['input_time_step']) + int(config['model']['output_time_step']) - 1    
    pred = np.concatenate([[[None]*pred.shape[1]]*tmp, pred], axis=0).transpose()
    
    pred_df = pd.DataFrame(pred_df.iloc[:,:10], columns=pred_df.columns[:10+71*72])
    
    print('Write prediction to DataFrame...')
    for i, col_name in tqdm(enumerate(pred_df.columns[10:])):
        pred_df[col_name] = pred[i]
    
    # post processing
    if config['preprocess']['predict_norm'] == 'True':
        pred_df.iloc[:,10:] = processer.postprocess(pred_df[:,10:])
        
    # rename dataframe
    pred_df = pred_df.rename(columns={'CODE':SWGIM_model_name})
    
    # rounding digit
    # pred_df = pred_df.round(rounding_digit)
    
    print(pred_df.info())
    print('Saving to csv file...')
    pred_df.to_csv(RECORDPATH / Path('prediction.csv')) # , float_format=f'%.{rounding_digit}f'