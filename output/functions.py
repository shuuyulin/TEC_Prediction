from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def SWGIM_export(config, pred, *args, **kwargs):
    o_step, i_step = int(config['model']['output_time_step']), int(config['model']['input_time_step'])
    
    if len(pred.shape) > 2: # sequence GTEC map output
        for cur_step in [3]:#range(o_step):
            pred_step = pred[:,cur_step]
            
            # pred_step = np.concatenate([[[None]*pred_step.shape[-1]]*(i_step + cur_step), pred_step[:-cur_step-1]], axis=0).transpose()
            new_pred = [[[None]*pred_step.shape[-1]]*(i_step + cur_step),
                        pred_step,
                        [[None]*pred_step.shape[-1]]*(o_step - cur_step - 1)] if o_step - cur_step - 1 != 0  \
                else   [[[None]*pred_step.shape[-1]]*(i_step + cur_step),
                        pred_step]
            pred_step = np.vstack(new_pred).transpose()
            print(cur_step, pred_step.shape)
            export_a_frame(config, pred_step, cur_step+1, *args, **kwargs)
    else:
        
        pred = np.concatenate([[[None]*pred.shape[-1]]*(i_step + o_step - 1), pred], axis=0).transpose()
        export_a_frame(config, pred, o_step, *args, **kwargs)
                    
def export_a_frame(config, pred, step, pred_df, processer, RECORDPATH, rounding_digit):
    
    SWGIM_model_name = config['model']['model_name'].split('_')[0]
    
    pred_df = pd.DataFrame(pred_df.iloc[:,:10], columns=pred_df.columns[:10+71*72])
    
    print('Write prediction to DataFrame...')
    for i, col_name in tqdm(enumerate(pred_df.columns[10:])):
        pred_df[col_name] = pred[i].tolist()
    
    # post processing
    if config['preprocess']['predict_norm'] == 'True':
        pred_df.iloc[:,10:] = processer.postprocess(pred_df[:,10:])
        
    # rename dataframe
    pred_df = pred_df.rename(columns={'CODE':SWGIM_model_name})
    
    # rounding digit
    pred_df = pred_df.round(rounding_digit)
    
    print(pred_df.info())
    print('Saving to csv file...')
    pred_df.to_csv(RECORDPATH / Path(f'prediction_frame{step}.csv')) # , float_format=f'%.{rounding_digit}f'