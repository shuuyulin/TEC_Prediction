import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Seq2Seq(nn.Module):
    def __init__(self, config, arg, input_dim, output_dim, criterion=None):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.criterion = criterion
        
        self.device = config['global']['device']
        self.input_time_step = config.getint('model', 'input_time_step')
        self.output_time_step = config.getint('model', 'output_time_step')
        self.embedding_size = config.getint('model', 'embedding_size')
        self.hidden_size = config.getint('model', 'hidden_size')
        self.num_layer = config.getint('model', 'num_layer')
        self.dropout = config.getfloat('model', 'dropout') if self.num_layer > 1 else 0
        
        self.encoder = Encoder(self.input_dim, self.output_dim, self.embedding_size, self.hidden_size, self.num_layer, self.dropout)
        self.decoder = Decoder(self.input_dim, self.output_dim, self.embedding_size, self.hidden_size, self.num_layer, self.dropout)

    def forward(self, x, y):
        
        # tensor to store decoder outputs of each time step
        outputs = torch.zeros(y.shape).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(x)

        # first input to decoder is last coordinates of x
        # only take TEC feature
        decoder_input = x[:, -1, -1].unsqueeze(-1).unsqueeze(-1)
        
        for i in range(self.output_time_step):
            # run decode for one time step
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            # print(outputs.shape)
            # print(output.shape)
            # place predictions in a tensor holding predictions for each time step
            outputs[:, i:i+1, :] = output
            
            # auto regression, avoid teacher forcing
            decoder_input = output
            
        return outputs[:,-1], self.criterion(outputs.float(), y.float())\
            if self.criterion else outputs[:,-1]
            # only takes last output time step

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_size, hidden_size, num_layer, dropout):
        super().__init__()
        
        self.embedding = nn.Linear(in_features=input_dim, out_features=embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                            num_layers=num_layer, batch_first=True, dropout=dropout)
        
    def forward(self, x):
        """
        x: input batch data, size: [batch size, sequence len, feature size]
        """
        embedded = self.dropout(F.relu(self.embedding(x)))
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_size, hidden_size, num_layer, dropout):
        super().__init__()
        self.embedding = nn.Linear(in_features=1, out_features=embedding_size)
        # only take TEC feature but SW feature
        # to ensure in_feature dim of embedding equals out_feature dim of fc
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                            num_layers=num_layer, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_dim) # regression

    def forward(self, x, hidden, cell):
        
        embedded = self.dropout(F.relu(self.embedding(x)))
        lstm_output, _ = self.lstm(embedded, (hidden, cell))
        
        # ==QUESTION== need activation function?
        pred = self.fc(lstm_output)

        return pred, hidden, cell
