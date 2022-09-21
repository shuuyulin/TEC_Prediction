import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformer(nn.Module):
    def __init__(self, config, feature_dim, criterion=None):
        super(Transformer, self).__init__()
        self.config = config
        self.feature_dim = feature_dim
        self.criterion = criterion
        
        self.device = config['global']['device']
        self.input_time_step = int(config['model']['input_time_step'])
        self.output_time_step = int(config['model']['output_time_step'])
        self.hidden_size = int(config['model']['hidden_size'])
        self.num_layer = int(config['model']['num_layer'])
        self.dropout = float(config['model']['dropout'])
        
        self.embedding = nn.Linear(feature_dim, self.hidden_size)
        self.pos_encoder = PositionalEncoding(d_model=self.hidden_size)
        self.transformer_model = nn.Transformer(d_model=self.hidden_size, nhead=8,\
                                                num_encoder_layers=self.num_layer,\
                                                num_decoder_layers=self.num_layer,\
                                                dropout=self.dropout, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 71*72) # global prediction
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.bias.data.zero_()
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)==1).transpose(0, 1))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, y):
        # x: (batch_size, input_time_step, feature_dim)
        # y: (batch_size, output_time_step, feature_dim)
        x_embedded = self.embedding(x)
        y_embedded = self.embedding(y)
        
        # x_mask = self._generate_square_subsequent_mask(x_embedded.size(1))
        y_mask = self._generate_square_subsequent_mask(y_embedded.size(1)).to(self.device)
        
        x_pos_encoder_out = self.pos_encoder(x_embedded)
        y_pos_encoder_out = self.pos_encoder(y_embedded)
        
        trans_output = self.transformer_model(src=x_pos_encoder_out, tgt=y_pos_encoder_out,\
                                            tgt_mask=y_mask)
        # trans_output: (batch_size, output_time_step, hidden_size)
        
        fc_out = self.fc(trans_output) # F.relu()
        # fc_out: (batch_size, output_time_step, 71*72)
        
        return fc_out[:,-1], self.criterion(fc_out.float(), y.float())\
                    if self.criterion != None else fc_out[:,-1]
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]