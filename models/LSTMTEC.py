import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMTEC(nn.Module):
    def __init__(self, config, feature_dim, criterion=None):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim
        self.criterion = criterion
        
        self.input_time_step = config.getint('model', 'input_time_step')
        self.output_time_step = config.getint('model', 'output_time_step')
        self.hidden_size = config.getint('model', 'hidden_size')
        self.lstm_layer = config.getint('model', 'lstm_layer')
        self.dropout = config.getfloat('model', 'dropout') if self.lstm_layer > 1 else 0
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=self.hidden_size,
                            num_layers=self.lstm_layer, dropout=self.dropout)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1) # regression

    def forward(self, feature_packed, truth_pad):
        
        lstm_output_packed, _ = self.lstm(feature_packed.float()) # batch, input_time_step, feature_dim
        # print(lstm_output_packed.shape)
        lstm_output, output_lengths = pad_packed_sequence(lstm_output_packed, batch_first=True)
        # print(lstm_output.shape)
        # ==QUESTION== need activation function?
        pred = self.fc(lstm_output[:,-1,:])

        
        return pred, self.criterion(pred.float(), truth_pad.float()) if self.criterion else pred