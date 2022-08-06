import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM_Seq2Seq(nn.Module):
    def __init__(self, config, feature_dim, criterion=None):
        super(LSTM_TEC, self).__init__()
        self.config = config
        self.feature_dim = feature_dim
        self.criterion = criterion
        
        self.device = config['global']['device']
        self.input_time_step = config.getint('model', 'input_time_step')
        self.output_time_step = config.getint('model', 'output_time_step')
        self.hidden_size = config.getint('model', 'hidden_size')
        self.lstm_layer = config.getint('model', 'lstm_layer')
        self.dropout = config.getfloat('model', 'dropout') if self.lstm_layer > 1 else 0
        
        self.encoder = Encoder(self.feature_dim, self.hidden_size, self.lstm_layer, self.dropout)
        self.decoder = Decoder(self.feature_dim, self.hidden_size, self.lstm_layer, self.dropout)

    def forward(self, feature_packed, truth_pad):
        
        # tensor to store decoder outputs of each time step
        outputs = torch.zeros(y.shape).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(feature_packed)

        # first input to decoder is last coordinates of feature_packed
        decoder_input = feature_packed[:, -1, :]
        
        for i in range(self.output_time_step):
            # run decode for one time step
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            # place predictions in a tensor holding predictions for each time step
            outputs[i] = output
            
            # Auto regression, avoid teacher forcing

        return outputs self.criterion(outputs.float(), truth_pad.float())\
            if self.criterion else outputs

class Encoder(nn.Module):
    def __init__(self, feature_dim, hidden_size, lstm_layer, dropout):
        super().__init__()
                
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size,
                            num_layers=lstm_layer, dropout=dropout)

    def forward(self, x):
        """
        x: input batch data, size: [sequence len, batch size, feature size]
        for the argoverse trajectory data, size(x) is [20, batch size, 2]
        """
        embedded = self.dropout(F.relu(self.linear(x)))
        output, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, feature_dim, hidden_size, num_layers, dropout):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size,
                            num_layers=lstm_layer, dropout=dropout)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1) # regression

    def forward(self, feature_packed, hidden, cell):
        
        lstm_output_packed, _ = self.lstm(feature_packed.float(), (hidden, cell))
        # print(lstm_output_packed.shape)
        lstm_output, output_lengths = pad_packed_sequence(lstm_output_packed, batch_first=True)
        # print(lstm_output.shape)
        # ==QUESTION== need activation function?
        pred = self.fc(lstm_output[:,-1,:])

        return pred, hidden, cell
