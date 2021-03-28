import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import EncodeAttentionQMnM1M2M3NoFF

class EncodeSmallAttentionQMnM1M2M3NoFF(EncodeAttentionQMnM1M2M3NoFF):
    def __init__(
        self, vocabs, dim, dropout, enc_layer, dec_layers, bidirectional, 
        mean, std, dec_in_avg, m_bool, layer_norm, max_len):
        super().__init__(
            vocabs, dim, dropout, enc_layer, dec_layers, bidirectional, 
            mean, std, dec_in_avg, m_bool, layer_norm, max_len)
    
    def init_mvals_network(self):
        pass

    def init_qvals_network(self):
        self.emb_ratings = nn.Embedding(num_embeddings=len(self.vocabs.ratings), embedding_dim=1)
        self.linear_q = nn.Linear(in_features=3, out_features=self.dim)
        self.lstm_enc = nn.LSTM(
            input_size=self.dim, hidden_size=self.dim, num_layers=self.enc_layers, batch_first=True, 
            dropout=self.p, bidirectional=self.bidirectional)
    
    def init_attention_network(self, dec_size, enc_q_size, enc_meta_size):
        super().init_attention_network(dec_size, enc_q_size, enc_meta_size)
        del self.linear_att_dec_m
        del self.linear_att_m
        del self.linear_att_dec_m_m
    
    def init_decoder_network(self):
        if self.bidirectional:
            hidden_size = self.dim * 2
        else:
            hidden_size = self.dim
        
        if self.dec_layers>1:
            self.lstm_decoder_input=nn.LSTM(
                input_size=self.dim, hidden_size=hidden_size, num_layers=self.dec_layers-1, 
                batch_first=True, dropout=self.p, bidirectional=False)
            self.lstm_dec = nn.LSTMCell(input_size=hidden_size*2 + self.dim*3, hidden_size=hidden_size)
            self.init_attention_network(hidden_size, hidden_size, self.dim)

        else:
            self.lstm_dec = nn.LSTMCell(input_size=self.dim*4 + hidden_size, hidden_size=hidden_size)
            self.init_attention_network(self.dim, hidden_size, self.dim)
        
        start_in = hidden_size*2 + self.dim*3
        self.linear_dec1 = nn.Linear(in_features=start_in, out_features=int(start_in/3))
        self.linear_dec2 = nn.Linear(in_features=int(start_in/3), out_features=1)
    
    def dims(self, length):
        out = []
        for i in range(length):
            out.append(i)
            out.append(i+length)
            out.append(i+length+length)
        
        return out
    
    def encode_qvals(self, qvals, ratings):
        bsize, length = ratings.shape
        emb = self.emb_ratings(ratings.type(torch.long)).reshape(bsize, length)
        x = torch.cat([qvals, emb], dim=1)[:, self.dims(length)].reshape(bsize, length, 3)
        x = self.dropout(self.linear_q(x))
        output, _ =self.lstm_enc(x)

        return output

    def do_encoding(self, batch):
        qvals = torch.cat([batch['qvals'] , batch['mvals']], dim=1)
        if self.dec_in_avg:
            enc_q = self.encode_qvals(qvals, batch['qvals_ratings'][:,:-1])
        else:
            enc_q = self.encode_qvals(qvals, batch['qvals_ratings'][:,:-2])
        
        enc_meta = self.encode_meta(
            torch.cat([batch['meta'], batch['qvals_ratings'][:,[-1]]], dim=1), batch['col2idx'])
        enc_meta2 = self.encode_meta2(batch['meta2'], batch['col2idx_2'])
        enc_meta3 = self.encode_meta3(batch['meta3'])

        return enc_q, enc_meta, enc_meta2, enc_meta3
    
    def init_lstm_dec_hidden(self, encoded_q, mask) -> torch.tensor:
        bsize, n = mask.shape
        output = encoded_q * mask.reshape(bsize, n, 1)
        output = torch.sum(output, dim=1, keepdim=False)/(torch.sum(mask, dim=1, keepdim=True) + 1e-10)

        return self.dropout(output)
    
    def enc_decoder_input(self, qval, rating, encoded_q, mask):
        emb = self.emb_ratings(rating.type(torch.long))
        _input = self.dropout(self.linear_q(torch.cat([qval,emb], dim=1)))
        hidden_init = self.init_lstm_dec_hidden(encoded_q, mask)

        if self.dec_layers==1:
            return _input, hidden_init
        else:
            bsize, dim = hidden_init.shape
            temp_hidden_init = torch.cat(
                [hidden_init for i in range(self.dec_layers-1)], dim=1).reshape(self.dec_layers-1, bsize, dim)
            output, _ = self.lstm_decoder_input(
                _input.reshape(bsize, 1, _input.shape[1]), (temp_hidden_init, temp_hidden_init))
            
            return output.reshape(bsize, dim), hidden_init

    def do_attention(self, batch, dec_emb, encoded_map):
        ctx_qvals = self.attention('qvals', encoded_map['qvals'], batch['enc_mask'], batch['enc_mask2'], dec_emb)
        ctx_meta = self.attention('meta', encoded_map['meta'], None, None, dec_emb)
        ctx_meta2 = self.attention('meta2', encoded_map['meta2'], None, None, dec_emb)
        ctx_meta3 = self.attention('meta3', encoded_map['meta3'], batch['meta3_mask'], batch['meta3_mask2'], dec_emb)

        return ctx_qvals, ctx_meta, ctx_meta2, ctx_meta3

    def forward(self, batch):
        enc_qvals, enc_meta, enc_meta2, enc_meta3 = self.do_encoding(batch)
        encoded_map = {'qvals': enc_qvals, 'meta': enc_meta, 'meta2': enc_meta2, 'meta3': enc_meta3}
        dec_emb, hidden_init = self.enc_decoder_input(
            torch.cat([batch['qvals_dec_in'], batch['mvals_dec_in']], dim=1), 
            batch['qvals_ratings'][:, -2], enc_qvals, batch['enc_mask'])
        ctx_qvals, ctx_meta, ctx_meta2, ctx_meta3 = self.do_attention(batch, dec_emb, encoded_map)

        # do decoding
        output, _ = self.lstm_dec(
            torch.cat([dec_emb, ctx_qvals, ctx_meta, ctx_meta2, ctx_meta3], dim=1), 
            (hidden_init, hidden_init)) 
        output = self.dropout(
            F.relu(self.linear_dec1(torch.cat([output, ctx_qvals, ctx_meta, ctx_meta2, ctx_meta3], dim=1)))) 
        output = self.linear_dec2(output)

        if(self.un_normalize):
            return self.do_unnorm(output)
        else:
            return output

