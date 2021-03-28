import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import layer_norm

from models.network import EncodeAttentionQnM1, FeedForward

class EncodeAttentionQMnM1M2(EncodeAttentionQnM1):
    def __init__(
        self, vocabs, dim, dropout, enc_layers, dec_layers, bidirectional, 
        mean, std, dec_in_avg, m_bool, layer_norm, max_len):
        super().__init__(
            vocabs, dim, dropout, enc_layers, dec_layers, bidirectional, 
            mean, std, dec_in_avg, m_bool, layer_norm, max_len)
        self.init_meta2_network()
        self.init_mvals_network()

        self.meta2_att = None
        self.m_att = None
    
    def init_mvals_network(self):
        self.linear_m = nn.Linear(in_features=1, out_features=self.dim)
        self.lstm_m = nn.LSTM(
            input_size=self.dim, hidden_size=self.dim, num_layers=self.enc_layers, batch_first=True, 
            dropout=self.p, bidirectional=self.bidirectional)
    
    def init_meta2_network(self):
        self.meta2_cols = ['faceno', 'rpa_op', 'rpa_type', 'rpa_inter', 'construct']
        col2vocab = {
            'faceno': self.vocabs.faceno, 'rpa_op': self.vocabs.rpa_op,
            'rpa_type': self.vocabs.rpa_type, 'rpa_inter': self.vocabs.rpa_inter,
            'construct': self.vocabs.construct}
        self.emb_meta2 = nn.ModuleDict({})
        self.ff_meta2 = nn.ModuleDict({})

        for i in range(len(self.meta2_cols)):
            self.emb_meta2[self.meta2_cols[i]] = nn.Embedding(
                num_embeddings=len(col2vocab[self.meta2_cols[i]]), embedding_dim=self.dim)
            self.ff_meta2[self.meta2_cols[i]] = FeedForward(
                self.dim, self.dim, self.enc_layers, self.dropout, self.layer_norm)
    
    def init_attention_network(self, dec_size, enc_size, enc_meta_size):
        super().init_attention_network(dec_size, enc_size, enc_meta_size)
        self.linear_att_meta2 = nn.Linear(in_features=enc_meta_size, out_features=enc_size, bias=False)
        self.linear_att_dec_meta2 = nn.Linear(in_features=enc_size, out_features=1, bias=True)
        self.linear_att_dec_m = nn.Linear(in_features=dec_size, out_features=enc_size, bias=False)
        self.linear_att_m = nn.Linear(in_features=enc_size, out_features=enc_size, bias=False)
        self.linear_att_dec_m_m = nn.Linear(in_features=enc_size, out_features=1, bias=True)
    
    def init_decoder_network(self):
        if self.bidirectional:
            hidden_size = self.dim * 2
        else:
            hidden_size = self.dim
        
        if self.dec_layers>1:
            self.lstm_decoder_input_m=nn.LSTM(
                input_size=self.dim, hidden_size=hidden_size, num_layers=self.dec_layers-1, 
                batch_first=True, dropout=self.p, bidirectional=False)
            self.lstm_dec = nn.LSTMCell(
                input_size=hidden_size*4 + self.dim*(2), 
                hidden_size=hidden_size)
            self.init_attention_network(hidden_size, hidden_size, self.dim)

        else:
            self.lstm_dec = nn.LSTMCell(
                input_size=self.dim*(4) + hidden_size*2, 
                hidden_size=hidden_size)
            self.init_attention_network(self.dim, hidden_size, self.dim)
        
        # final output layer
        start_in = hidden_size*3 + self.dim*(2)
        self.linear_dec1 = nn.Linear(in_features=start_in, out_features=int(start_in/3))
        self.linear_dec2 = nn.Linear(in_features=int(start_in/3), out_features=1)

    
    def init_lstm_dec_hidden(self, encoded_q, mask):
        bsize, n = mask.shape
        output = encoded_q * mask.reshape(bsize, n, 1)
        output = torch.sum(output, dim=1, keepdim=False)/(torch.sum(mask, dim=1, keepdim=True) + 1e-10)

        return output
    
    def encode_mvals(self, mvals):
        bsize, dim = mvals.shape
        output, _ = self.lstm_m(self.dropout(self.linear_m(mvals.reshape(bsize, dim, 1))))

        return output

    def enc_decoder_input_m(self, mvals, encoded_m, mask):
        _input = self.dropout(self.linear_m(mvals))
        hidden_init = self.init_lstm_dec_hidden(encoded_m, mask)

        if self.dec_layers==1:
            return _input, hidden_init
        else:
            bsize, dim = hidden_init.shape
            temp_hidden_init = torch.cat(
                [hidden_init for i in range(self.dec_layers-1)], dim=1).reshape(self.dec_layers-1, bsize, dim)
            output, _ = self.lstm_decoder_input_m(
                _input.reshape(bsize, 1, _input.shape[1]), (temp_hidden_init, temp_hidden_init))
            
            return output.reshape(bsize, dim), hidden_init
    
    def attention(self, ds, encoded, mask, mask2, dec_hidden):
        out = super().attention(ds, encoded, mask, mask2, dec_hidden)

        if out is not None:
            return out

        elif ds=='mvals':
            s = self.linear_att_dec_m(dec_hidden)
            h = self.linear_att_m(encoded)
            scores = self.linear_att_dec_m_m(torch.tanh(s.reshape(s.shape[0], 1, s.shape[1]) + h))
            if self.m_bool:
                scores.masked_fill_((1-mask.unsqueeze(-1)).type(torch.bool), -1e8)
            else:
                scores.masked_fill_((1-mask.unsqueeze(-1)).byte(), -1e8)
            weights = F.softmax(scores, dim=1)
            weights = weights * (mask2.unsqueeze(-1))
            out = torch.sum(weights * encoded, dim=1, keepdim=False)
            self.m_att = weights

            return out

        elif ds=='meta2':
            s = self.linear_att_dec(dec_hidden)
            h = self.linear_att_meta2(encoded)
            scores = self.linear_att_dec_meta2(torch.tanh(s.reshape(s.shape[0], 1, s.shape[1]) + h))
            weights = F.softmax(scores, dim=1)
            out = torch.sum(weights * encoded, dim=1, keepdim=False)
            self.meta2_att = weights

            return out

        else:
            return None
    
    def do_encoding(self, batch):
        enc_qvals, enc_meta = super().do_encoding(batch)
        enc_meta2 = self.encode_meta2(batch['meta2'], batch['col2idx_2'])
        enc_mvals = self.encode_mvals(batch['mvals'])

        return enc_qvals, enc_mvals, enc_meta, enc_meta2
    
    def do_encoding_dec_inputs(self, batch, encoded_map):
        dec_in_qvals, hidden_init_qvals = self.enc_decoder_input(
            batch['qvals_dec_in'], batch['qvals_ratings'][:, -2], encoded_map['qvals'], 
            batch['qvals_mask'])
        dec_in_mvals, hidden_init_mvals = self.enc_decoder_input_m(
            batch['mvals_dec_in'], encoded_map['mvals'], batch['mvals_mask'])
        hidden_init = self.dropout((hidden_init_qvals + hidden_init_mvals)/2)

        return dec_in_qvals, dec_in_mvals, hidden_init
    
    def do_attention(self, batch, dec_in_map, encoded_map):
        ctx_qvals, ctx_meta = super().do_attention(batch, dec_in_map['qvals'], encoded_map)
        ctx_mvals = self.attention(
            'mvals', encoded_map['mvals'], batch['mvals_mask'], batch['mvals_mask2'], 
            dec_in_map['mvals'])
        ctx_meta2 = self.attention('meta2', encoded_map['meta2'], None, None, dec_in_map['qvals'])

        return ctx_qvals, ctx_mvals, ctx_meta, ctx_meta2

    def forward(self, batch):
        enc_qvals, enc_mvals, enc_meta, enc_meta2 = self.do_encoding(batch)
        encoded_map = {'qvals': enc_qvals, 'mvals': enc_mvals, 'meta': enc_meta, 'meta2': enc_meta2}
        dec_in_qvals, dec_in_mvals, hidden_init = self.do_encoding_dec_inputs(batch, encoded_map)
        dec_in_map = {'qvals': dec_in_qvals, 'mvals': dec_in_mvals}
        ctx_qvals, ctx_mvals, ctx_meta, ctx_meta2 = self.do_attention(batch, dec_in_map, encoded_map)
        
        # do decoding
        output, _ = self.lstm_dec(
            torch.cat([dec_in_qvals, dec_in_mvals, ctx_qvals, ctx_mvals, ctx_meta, ctx_meta2], dim=1), 
            (hidden_init, hidden_init)) 
        output = self.dropout(
            F.relu(self.linear_dec1(torch.cat([output, ctx_qvals, ctx_mvals, ctx_meta, ctx_meta2], dim=1)))) 
        output = self.linear_dec2(output)

        if(self.un_normalize):
            return self.do_unnorm(output)
        else:
            return output