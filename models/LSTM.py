import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class single_point_LSTM(nn.Module):
    def __init__(self, config, criterion=None):
        super(single_point_LSTM, self).__init__()
        self.config = config
        self.criterion = criterion
        
        self.predict_met = config['model']['predict_met']
        self.input_time_step = config.getint('model', 'input_time_step')
        self.output_time_step = config.getint('model', 'output_time_step')
        self.hidden_size = config.getint('model', 'hidden_size')
        self.lstm_layer = config.getint('model', 'lstm_layer')
        self.dropout = config.getfloat('model', 'dropout') if self.lstm_layer > 1 else 0
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size,
                            num_layers=self.lstm_layer, dropout=self.dropout)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1) # regression

    def forward(self, tec_packed, truth_pad):
        
        if self.predict_met == 'continuous':
            output, _ = self.lstm(tec_packed)
            # ==TODO== iterate 4 times
        elif self.predict_met == 'incontinuous':
            # print(f'input: {tec_packed.is_cuda}')
            lstm_output_packed, _ = self.lstm(tec_packed.float())
            # print(lstm_output_packed.shape)
            lstm_output, output_lengths = pad_packed_sequence(lstm_output_packed, batch_first=True)
            # print(lstm_output.shape)
            # ==QUESTION== need activation function?
            pred = self.fc(lstm_output[:,-1,:])

        return pred, self.criterion(pred.float(), truth_pad.float()) if self.criterion else pred