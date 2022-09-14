# TEC_Prediction

## Usage
```
python3 main.py --config [config_file_path] --mode [mode train/test] --record [record_path]
```
## File explanation

- config.ini  
  - config file, containing all configurations  
- main.py  
  - main func: main loop, initialize all things and run epoches  
  - train_one func: train one epoch  
  - eval_one func: evaluate one epoch and record outputs  
- dataset
  - initialize dataloader
  - collate_fn.py  
    - initialize collate_fn to adjust batched data  
- models
  - initialize model
- output
  - ignore it  :)

- preprocessing
  - initialize normalization

- training_tools
  - initialize optimizer, lr_scheduler and criterion
