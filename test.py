import torch.nn as nn
import torch
from collections import OrderedDict

def setSeed(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
setSeed(31)    

class GetLSTMOutput(nn.Module):
    def forward(self, x):
        out, _ = x
        return out
    
input_size, output_size = 10, 2
hidden_size = 3
seq_len = 3
rnn1 = nn.LSTM(input_size, hidden_size, 2)

rnn2 = nn.Sequential(OrderedDict([
    ('LSTM1', nn.LSTM(input_size, hidden_size, 1)),
    ('out', GetLSTMOutput()),
    ('LSTM2', nn.LSTM(hidden_size, hidden_size, 1)),
    # ('out', GetLSTMOutput()),
]))

input = torch.randn(seq_len, 5, input_size)
# print(input)
print(rnn1(input)[0]) # get output
print(rnn2(input)[0]) # get output

