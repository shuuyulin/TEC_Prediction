from models.Transformer import Transformer
import torch
import torch.nn as nn
import pandas as pd
config = {
    'global' : {'device':'cpu'},
    'model' : {'input_time_step':24,
             'output_time_step':4,
             'hidden_size':384,
             'num_layer':1,
             'dropout':0.1,
             },
}
criterion = nn.MSELoss()
model = Transformer(config, 71*72, criterion)
model.load_state_dict(torch.load('./record/29/best_model.pth'))

test_df = pd.read_csv('./data/raw_data/SWGIM_day/2015/2015001.csv', header=list(range(6)), index_col=0)

input = test_df[('CODE', 'GIM')]
input = torch.tensor(input.values)
print(input.shape)

output = model(input)
print(output.shape)



