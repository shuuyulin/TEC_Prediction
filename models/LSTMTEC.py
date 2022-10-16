import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMTEC(nn.Module):
    def __init__(self, config, arg, input_dim, output_dim, criterion=None):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.criterion = criterion
        
        self.input_time_step = config.getint('model', 'input_time_step')
        self.output_time_step = config.getint('model', 'output_time_step')
        self.hidden_size = config.getint('model', 'hidden_size')
        self.num_layer = config.getint('model', 'num_layer')
        self.dropout = config.getfloat('model', 'dropout') if self.num_layer > 1 else 0
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layer, dropout=self.dropout)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_dim) # regression

    def forward(self, feature_packed, truth_pad):
        
        lstm_output_packed, _ = self.lstm(feature_packed.float()) # batch, input_time_step, input_dim
        # print(lstm_output_packed.shape)
        lstm_output, output_lengths = pad_packed_sequence(lstm_output_packed, batch_first=True)
        # print(lstm_output.shape)
        # ==QUESTION== need activation function?
        pred = self.fc(lstm_output[:,-1,:])

        
        return pred, self.criterion(pred.float(), truth_pad.float()) if self.criterion else pred