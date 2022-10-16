import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformer(nn.Module):
    def __init__(self, config, arg, input_dim, output_dim, criterion=None):
        super(Transformer, self).__init__()
        self.config = config
        self.arg = arg
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.criterion = criterion
        
        self.device = config['global']['device']
        self.input_time_step = int(config['model']['input_time_step'])
        self.output_time_step = int(config['model']['output_time_step'])
        self.hidden_size = int(config['model']['hidden_size'])
        self.num_layer = int(config['model']['num_layer'])
        self.dropout = float(config['model']['dropout'])
        
        self.embedding = nn.Linear(input_dim, self.hidden_size)
        self.pos_encoder = PositionalEncoding(d_model=self.hidden_size)
        self.transformer_model = nn.Transformer(d_model=self.hidden_size, nhead=8,\
                                                num_encoder_layers=self.num_layer,\
                                                num_decoder_layers=self.num_layer,\
                                                dropout=self.dropout, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, output_dim) # global prediction
        self.init_weights()
           
    # def _model_forward(self, src, tgt):
    def forward(self, x, y):
        # x: (batch_size, input_time_step, input_dim)
        # y: (batch_size, output_time_step, input_dim)
        bs, fd = x.shape[0], x.shape[2]
        BOS = torch.full((bs, 1, fd), 0, dtype=torch.float).to(self.device)
        
        src = x
        src_embedded = self.embedding(src)
        src_pe_out = self.pos_encoder(src_embedded)
        memory = self.transformer_model.encoder(src_pe_out)
        
        tgt = BOS
        for _ in range(self.output_time_step):
            
            tgt_embedded = self.embedding(tgt)
            tgt_mask = self._generate_square_subsequent_mask(tgt_embedded.size(1)).to(self.device)
            tgt_pe_out = self.pos_encoder(tgt_embedded)
            
            trans_out = self.transformer_model.decoder(tgt_pe_out, memory,\
                                                tgt_mask=tgt_mask)
            # trans_out: (batch_size, output_time_step, hidden_size)
            
            fc_out = self.fc(trans_out) # F.relu()
            # fc_out: (batch_size, output_time_step, 71*72)
            tgt = torch.cat((tgt, fc_out[:,-1:]), dim=1)
            
        return fc_out[:,-1], self.criterion(fc_out.float(), y.float())\
                    if self.criterion != None else fc_out[:,-1]
            
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