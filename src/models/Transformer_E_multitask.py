import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import wandb
class Transformer_E_mttasks(nn.Module):
    def __init__(self, config, shaper, criterion=None):
        super(Transformer_E_mttasks, self).__init__()
        self.config = config
        self.shaper = shaper
        self.criterion = criterion
        input_dim = self.shaper.get_input_dim()
        tec_output_dim = self.shaper.get_tec_output_dim()
        other_output_dim = self.shaper.get_other_output_dim()
        
        self.device = config['global']['device']
        self.input_time_step = config.getint('model', 'input_time_step')
        self.output_time_step = config.getint('model', 'output_time_step')
        self.hidden_size = config.getint('model', 'hidden_size')
        self.num_layer = config.getint('model', 'num_layer')
        self.dropout = config.getfloat('model', 'dropout')
        
        self.embedding = nn.Linear(input_dim, self.hidden_size)
        self.pos_encoder = PositionalEncoding(d_model=self.hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8,\
                            dropout=self.dropout, norm_first=True, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layer)
        self.tec_predictor = nn.Linear(self.hidden_size, tec_output_dim)
        self.other_predictor = nn.Linear(self.hidden_size, other_output_dim)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.bias.data.zero_()
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.tec_predictor.bias.data.zero_()
        self.tec_predictor.weight.data.uniform_(-initrange, initrange)
        self.other_predictor.bias.data.zero_()
        self.other_predictor.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)==1).transpose(0, 1))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, **y):
        # x: (batch_size, input_d)
        # print(x.shape)
        embedded = self.embedding(x)
        pos_encoder_out = self.pos_encoder(embedded)
        # mask = self._generate_square_subsequent_mask(pos_encoder_out.size(1)).to(self.device)
        # trans_output = self.transformer_encoder(pos_encoder_out, mask) # batch_size, seq_len, hidden_size
        trans_output = self.transformer_encoder(pos_encoder_out) # batch_size, seq_len, hidden_size
        # trans_output = trans_output # batch_size, seq_len, hidden_size
        
        tec_out = F.relu(self.tec_predictor(trans_output)) # batch_size, _, _
        other_out = self.other_predictor(trans_output)
        
        tec_pred = self.shaper.model_tec_drop(tec_out) # batch_size, 71*72
        other_pred = self.shaper.model_other_drop(other_out) # batch_size, n_ft_out
        #TODO: possible error
        # print(tec_pred.shape)
        # print(torch.swapaxes(other_pred.unsqueeze(-1), 0, 1).shape)
        # print(list(y.values())[1:][0].shape)
        loss, losses = self.criterion((tec_pred, y['tec']), *zip(torch.swapaxes(other_pred.unsqueeze(-1), 0, 1), list(y.values())[1:]))
        return tec_pred, {'loss':loss, 'losses':losses}\
            if self.criterion != None else tec_pred
    
    def record(self, loss, mode):
        wandb.log({f'{mode}_loss':loss['loss'], f'{mode}_tec_loss':loss['losses'][0], f'{mode}_dst_loss':loss['losses'][1]})
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
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