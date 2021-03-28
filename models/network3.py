import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.network2 import EncodeAttentionQMnM1M2

class EncodeAttentionQMQnMM(EncodeAttentionQMnM1M2):
    def __init__(
        self, vocabs, dim, dropout, enc_layer, dec_layers, bidirectional, 
        mean, std, dec_in_avg, m_bool, layer_norm, max_len):
        super().__init__(vocabs, dim, dropout, enc_layer, dec_layers, bidirectional, 
        mean, std, dec_in_avg, m_bool, layer_norm, max_len)
        self.init_qic_network()
        self.qic_att = None

    def init_qic_network(self):
        self.linear_qic = nn.Linear(in_features=1, out_features=self.dim)
        self.lstm_qic = nn.LSTM(
            input_size=self.dim, hidden_size=self.dim, num_layers=self.enc_layers, batch_first=True, 
            dropout=self.p, bidirectional=self.bidirectional)
    
    def init_attention_network(self, dec_size, enc_lstm_size, enc_meta_size):
        super().init_attention_network(dec_size, enc_lstm_size, enc_meta_size)
        self.linear_att_qic = nn.Linear(in_features=enc_lstm_size, out_features=enc_lstm_size)
        self.linear_att_dec_qic = nn.Linear(in_features=dec_size, out_features=enc_lstm_size, bias=False)
        self.linear_att_dec_qic_qic = nn.Linear(in_features=enc_lstm_size, out_features=1, bias=True)
    
    def init_decoder_network(self):
        if self.bidirectional:
            hidden_size = self.dim * 2
        else:
            hidden_size = self.dim
        
        if self.dec_layers>1:
            self.lstm_decoder_input_m=nn.LSTM(
                input_size=self.dim, hidden_size=hidden_size, num_layers=self.dec_layers-1, 
                batch_first=True, dropout=self.p, bidirectional=False)
            self.lstm_decoder_input_qic=nn.LSTM(
              input_size=self.dim, hidden_size=hidden_size, num_layers=self.dec_layers-1, 
                batch_first=True, dropout=self.p, bidirectional=False)
            self.lstm_dec = nn.LSTMCell(
                input_size=hidden_size*6 + self.dim*2, 
                hidden_size=hidden_size)
            self.init_attention_network(hidden_size, hidden_size, self.dim)

        else:
            self.lstm_dec = nn.LSTMCell(
                input_size=self.dim*5 + hidden_size*3, 
                hidden_size=hidden_size)
            self.init_attention_network(self.dim, hidden_size, self.dim)
        
        # final output layer
        start_in = hidden_size*4 + self.dim*(2)
        self.linear_dec1 = nn.Linear(in_features=start_in, out_features=int(start_in/3))
        self.linear_dec2 = nn.Linear(in_features=int(start_in/3), out_features=1)
    
    def do_encoding(self, batch):
        enc_qvals, enc_mvals, enc_meta, enc_meta2 = super().do_encoding(batch)
        bsize, dim = batch['qic'].shape
        enc_qic, _ = self.lstm_qic(self.dropout(self.linear_qic(batch['qic'].reshape(bsize, dim, 1))))

        return enc_qvals, enc_mvals, enc_qic, enc_meta, enc_meta2
    
    def enc_decoder_input_qic(self, qic, encoded_qic, mask):
        _input = self.dropout(self.linear_qic(qic))
        hidden_init = self.init_lstm_dec_hidden(encoded_qic, mask)

        if self.dec_layers==1:
            return _input, hidden_init
        else:
            bsize, dim = hidden_init.shape
            temp_hidden_init = torch.cat(
                [hidden_init for i in range(self.dec_layers-1)], dim=1).reshape(self.dec_layers-1, bsize, dim)
            output, _ = self.lstm_decoder_input_qic(
               _input.reshape(bsize, 1, _input.shape[1]), (temp_hidden_init, temp_hidden_init))
            
            return output.reshape(bsize, dim), hidden_init
    
    def do_encoding_dec_inputs(self, batch, encoded_map):
        dec_in_qvals, hidden_init_qvals = self.enc_decoder_input(
            batch['qvals_dec_in'], batch['qvals_ratings'][:, -2], encoded_map['qvals'], batch['qvals_mask'])
        dec_in_mvals, hidden_init_mvals = self.enc_decoder_input_m(
            batch['mvals_dec_in'], encoded_map['mvals'], batch['mvals_mask'])
        dec_in_qic, hidden_init_qic = self.enc_decoder_input_qic(
            batch['qic_dec_in'], encoded_map['qic'], batch['qic_mask'])
        hidden_init = self.dropout((hidden_init_mvals + hidden_init_qic + hidden_init_qvals)/3)

        return dec_in_qvals, dec_in_mvals, dec_in_qic, hidden_init

    def attention(self, ds, encoded, mask, mask2, dec_hidden):
        out = super().attention(ds, encoded, mask, mask2, dec_hidden)

        if out is not None:
            return out

        elif ds=='qic':
            s = self.linear_att_dec_qic(dec_hidden)
            h = self.linear_att_qic(encoded)
            scores = self.linear_att_dec_qic_qic(torch.tanh(s.reshape(s.shape[0], 1, s.shape[1]) + h))
            if self.m_bool:
                scores.masked_fill_((1-mask.unsqueeze(-1)).type(torch.bool), -1e8)
            else:
                scores.masked_fill_((1-mask.unsqueeze(-1)).byte(), -1e8)
            weights = F.softmax(scores, dim=1)
            weights = weights * (mask2.unsqueeze(-1))
            out = torch.sum(weights * encoded, dim=1, keepdim=False)
            self.m_att = weights

            return out

        else:
            return None
    def do_attention(self, batch, dec_in_map, encoded_map):
        ctx_qvals, ctx_mvals, ctx_meta, ctx_meta2 = super().do_attention(batch, dec_in_map, encoded_map)
        ctx_qic = self.attention(
            'qic', encoded_map['qic'], batch['qic_mask'], batch['qic_mask2'], dec_in_map['qic'])
        
        return ctx_qvals, ctx_mvals, ctx_qic, ctx_meta, ctx_meta2

    def forward(self, batch):
        enc_qvals, enc_mvals, enc_qic, enc_meta, enc_meta2 = self.do_encoding(batch)
        encoded_map = {'qvals': enc_qvals, 'mvals': enc_mvals, 'meta': enc_meta, 'meta2': enc_meta2, 'qic': enc_qic}
        dec_in_qvals, dec_in_mvals, dec_in_qic, hidden_init = self.do_encoding_dec_inputs(batch, encoded_map)
        dec_in_map = {'qvals': dec_in_qvals, 'mvals': dec_in_mvals, 'qic': dec_in_qic}
        ctx_qvals, ctx_mvals, ctx_qic, ctx_meta, ctx_meta2 = self.do_attention(batch, dec_in_map, encoded_map)

        # do decoding
        output, _ = self.lstm_dec(
            torch.cat([dec_in_qvals, dec_in_mvals, dec_in_qic, ctx_qvals, ctx_mvals, ctx_qic, ctx_meta, ctx_meta2], dim=1), 
            (hidden_init, hidden_init)) 
        output = self.dropout(
            F.relu(self.linear_dec1(torch.cat([output, ctx_qvals, ctx_mvals, ctx_qic, ctx_meta, ctx_meta2], dim=1)))) 
        output = self.linear_dec2(output)

        if(self.un_normalize):
            return self.do_unnorm(output)
        else:
            return output