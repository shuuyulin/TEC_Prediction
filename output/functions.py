from pathlib import Path
import numpy as np
import pandas as pd

def SWGIM_export(config, rounding_digit, pred, df, processer, RECORDPATH):
    
    tmp = max(int(config['data']['reserved']), int(config['model']['input_time_step']) + int(config['model']['output_time_step'])-1)
    SWGIM_model_name = config['model']['model_name'].split('_')[0]
    
    pred_df = pd.DataFrame([], columns=df.columns, index=df.index)
    pred_df['UTC'] = df['UTC']
    pred_df['OMNIWeb'] = df['OMNIWeb']
    
    pred = np.round_(pred, decimals=rounding_digit)
    
    pred = np.concatenate([[[None]*pred.shape[1]]*tmp, pred], axis=0)
    
    print('Write prediction to DataFrame...')
    pred_df.iloc[:,8:] = pred
    
    # post processing
    if config['preprocess']['predict_norm'] == 'True':
        pred_df = processer.postprocess(pred_df) # Possible error: np.nan denormed
        
    pred_df = pred_df.rename(columns={'CODE':SWGIM_model_name})
    
    print('Saving to csv file...')
    pred_df.to_csv(RECORDPATH / Path('prediction.csv'), float_format=f'%.{rounding_digit}f')