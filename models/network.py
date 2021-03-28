import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import layer_norm

from datasets import Vocabs

class FeedForward(nn.Module):
    def __init__(self, in_features, out_features, layers, dropout, do_layer_norm):
        super(FeedForward, self).__init__()
        self.layers = layers
        self.do_layer_norm = do_layer_norm
        self.linears = nn.ModuleList([nn.Linear(in_features, out_features) for i in range(layers)])

        if isinstance(dropout, float):
            self.dropout = nn.Dropout(dropout)
        elif isinstance(dropout, nn.Module):
            self.dropout = dropout
        else:
            raise ValueError

        if self.do_layer_norm:
            self.layernorms = nn.ModuleList([nn.LayerNorm([out_features]) for i in range(layers)])
        
    def forward_layer_norm(self, x):
        for i in range(self.layers-1):
            x = self.dropout(F.relu(self.layernorms[i](self.linears[i](x))))
        
        return F.relu(self.layernorms[-1](self.linears[-1](x)))

    def forward_no_layer_norm(self, x):
        for i in range(self.layers-1):
            x = self.dropout(F.relu(self.linears[i](x)))
        
        return F.relu(self.linears[-1](x))

    def forward(self, x):
        if self.do_layer_norm:
            return self.forward_layer_norm(x)
        else:
            return self.forward_no_layer_norm(x)

class EncodeAttentionQnM1(nn.Module):
    def __init__(
        self, vocabs: Vocabs, dim, dropout, enc_layers, dec_layers, bidirectional, 
        mean, std, dec_in_avg, m_bool, layer_norm, max_len):
        super(EncodeAttentionQnM1, self).__init__()
        self.mean = mean
        self.std = std
        self.dec_in_avg = dec_in_avg
        self.m_bool = m_bool
        self.vocabs = vocabs
        self.dim = dim
        self.p = dropout
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.bidirectional = bidirectional
        self.layer_norm = layer_norm
        self.max_len = max_len

        self.dropout = nn.Dropout(p=self.p)
        self.init_qvals_network()
        self.init_meta_network()
        self.init_decoder_network()

        self.un_normalize = False
        self.q_att = None
        self.meta_att = None
    
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def init_qvals_network(self):
        self.emb_ratings = nn.Embedding(num_embeddings=len(self.vocabs.ratings), embedding_dim=1)
        self.linear_q = nn.Linear(in_features=2, out_features=self.dim)
        self.lstm_enc = nn.LSTM(
            input_size=self.dim, hidden_size=self.dim, num_layers=self.enc_layers, batch_first=True, 
            dropout=self.p, bidirectional=self.bidirectional)
    
    def init_meta_network(self):
        self.emb_base = nn.Embedding(num_embeddings=len(self.vocabs.base), embedding_dim=self.dim)
        self.emb_region = nn.Embedding(num_embeddings=len(self.vocabs.region), embedding_dim=self.dim)
        self.emb_hq = nn.Embedding(num_embeddings=len(self.vocabs.hq), embedding_dim=self.dim)
        self.linear_rating = nn.Linear(in_features=1, out_features=self.dim)
        self.linear_year = nn.Linear(in_features=1, out_features=self.dim)

        self.ff_base = FeedForward(self.dim, self.dim, self.enc_layers, self.dropout, self.layer_norm)
        self.ff_region = FeedForward(self.dim, self.dim, self.enc_layers, self.dropout, self.layer_norm)
        self.ff_hq = FeedForward(self.dim, self.dim, self.enc_layers, self.dropout, self.layer_norm)
        self.ff_year = FeedForward(self.dim, self.dim, self.enc_layers,  self.dropout, self.layer_norm)
        self.ff_rating = FeedForward(self.dim, self.dim, self.enc_layers, self.dropout, self.layer_norm)
    
    def init_attention_network(self, dec_size, enc_q_size, enc_meta_size):
        self.linear_att_dec = nn.Linear(in_features=dec_size, out_features=enc_q_size, bias=False)
        self.linear_att_q = nn.Linear(in_features=enc_q_size, out_features=enc_q_size, bias=False)
        self.linear_att_meta = nn.Linear(in_features=enc_meta_size, out_features=enc_q_size, bias=False)
        self.linear_att_dec_q = nn.Linear(in_features=enc_q_size, out_features=1, bias=True)
        self.linear_att_dec_meta = nn.Linear(in_features=enc_q_size, out_features=1, bias=True)

    def init_decoder_network(self):
        if self.bidirectional:
            hidden_size = self.dim * 2
        else:
            hidden_size = self.dim
            
        if self.dec_layers>1:
            self.lstm_decoder_input=nn.LSTM(
                input_size=self.dim, hidden_size=hidden_size, num_layers=self.dec_layers-1, 
                batch_first=True, dropout=self.p, bidirectional=False)
            self.lstm_dec = nn.LSTMCell(input_size=hidden_size*2 + self.dim, hidden_size=hidden_size)
            self.init_attention_network(hidden_size, hidden_size, self.dim)

        else:
            self.lstm_dec = nn.LSTMCell(input_size=self.dim*2 + hidden_size, hidden_size=hidden_size)
            self.init_attention_network(self.dim, hidden_size, self.dim)
        
        start_in = hidden_size*2 + self.dim
        self.linear_dec1 = nn.Linear(in_features=start_in, out_features=int(start_in/3))
        self.linear_dec2 = nn.Linear(in_features=int(start_in/3), out_features=1)
        

    def dims(self, length):
        out = []
        for i in range(length):
            out.append(i)
            out.append(i + length)
        
        return out

    def encode_qvals(self, qvals, ratings):
        bsize, length = qvals.shape
        emb = self.emb_ratings(ratings.type(torch.long)).reshape(bsize, length)
        x = torch.cat([qvals, emb], dim=1)[:, self.dims(length)].reshape(bsize, length, 2)
        x = self.dropout(self.linear_q(x))
        output, _= self.lstm_enc(x)

        return output
    
    def encode_meta(self, meta, col2idx):
        bsize, _ = meta.shape
        enc_base = self.ff_base(
            self.dropout(self.emb_base(meta[:, col2idx['base']].type(torch.long))))
        enc_ratings = self.ff_rating(
            self.dropout(self.linear_rating(self.emb_ratings(meta[:, col2idx['rating']].type(torch.long)))))
        enc_region = self.ff_region(
            self.dropout(self.emb_region(meta[:, col2idx['region']].type(torch.long))))
        enc_hq = self.ff_hq(
            self.dropout(self.emb_hq(meta[:, col2idx['hq']].type(torch.long))))
        enc_year = self.ff_year(
            self.dropout(self.linear_year(meta[:, [col2idx['year']]])))

        return torch.cat(
            [enc_year, enc_hq, enc_region, enc_base, enc_ratings], dim=1).reshape(bsize, 5, self.dim)
    
    def encode_meta2(self, meta, col2idx):
        bsize, n = meta.shape
        encs = []

        for i in range(len(self.meta2_cols)):
            encs.append(self.ff_meta2[self.meta2_cols[i]](
                    self.dropout(self.emb_meta2[self.meta2_cols[i]](meta[:, i]))))
        
        return torch.cat(encs, dim=1).reshape(bsize, n, self.dim)
    
    def init_lstm_dec_hidden(self, encoded_q, mask) -> torch.tensor:
        bsize, n = mask.shape
        output = encoded_q * mask.reshape(bsize, n, 1)
        output = torch.sum(output, dim=1, keepdim=False)/(torch.sum(mask, dim=1, keepdim=True) + 1e-10)

        return self.dropout(output)

    def enc_decoder_input(self, qval, rating, encoded_q, mask):
        emb = self.emb_ratings(rating.type(torch.long))
        _input = self.dropout(self.linear_q(torch.cat([qval, emb], dim=1)))
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

    def attention(self, ds, encoded, mask, mask2, dec_hidden):
        if ds=='qvals':
            s = self.linear_att_dec(dec_hidden)
            h = self.linear_att_q(encoded)
            scores = self.linear_att_dec_q(torch.tanh(s.reshape(s.shape[0], 1, s.shape[1]) + h))
            if self.m_bool:
                scores.masked_fill_((1-mask.unsqueeze(-1)).type(torch.bool), -1e8)
            else:
                scores.masked_fill_((1-mask.unsqueeze(-1)).byte(), -1e8)
            weights = F.softmax(scores, dim=1)
            weights = weights * (mask2.unsqueeze(-1))
            out = torch.sum(weights * encoded, dim=1, keepdim=False)
            self.q_att = weights
            
            return out

        elif ds=='meta':
            s = self.linear_att_dec(dec_hidden)
            h = self.linear_att_meta(encoded)
            scores = self.linear_att_dec_meta(torch.tanh(s.reshape(s.shape[0], 1, s.shape[1]) + h))
            weights = F.softmax(scores, dim=1)
            out = torch.sum(weights * encoded, dim=1, keepdim=False)
            self.meta_att = weights

            return out
            
        else:
            return None
    
    def do_unnorm(self, output):
        return (output * self.std) + self.mean
    
    def do_encoding(self, batch):
        if self.dec_in_avg:
            enc_q = self.encode_qvals(batch['qvals'], batch['qvals_ratings'][:,:-1])
        else:
            enc_q = self.encode_qvals(batch['qvals'], batch['qvals_ratings'][:,:-2])
        
        enc_meta = self.encode_meta(
            torch.cat([batch['meta'], batch['qvals_ratings'][:, [-1]]], dim=1), batch['col2idx'])
        
        return enc_q, enc_meta
    
    def do_attention(self, batch, dec_emb, encoded_map):
        ctx_q = self.attention(
            'qvals', encoded_map['qvals'], batch['qvals_mask'], batch['qvals_mask2'], dec_emb)
        ctx_meta = self.attention('meta', encoded_map['meta'], None, None, dec_emb)

        return ctx_q, ctx_meta

    def forward(self, batch):
        enc_q, enc_meta = self.do_encoding(batch)
        dec_emb, hidden_init = self.enc_decoder_input(
            batch['qvals_dec_in'], batch['qvals_ratings'][:, -2], enc_q, batch['qvals_mask'])  
        ctx_q, ctx_meta = self.do_attention(batch, dec_emb, {'qvals': enc_q, 'meta': enc_meta})

        # do decoding
        output, _ = self.lstm_dec(
            torch.cat([dec_emb, ctx_q, ctx_meta], dim=1), (hidden_init, hidden_init))
        output = self.dropout(
            F.relu(self.linear_dec1(torch.cat([output, ctx_q, ctx_meta], dim=1))))
        output = self.linear_dec2(output)

        if(self.un_normalize):
            return self.do_unnorm(output)
        else:
            return output

class Average(EncodeAttentionQnM1):
    def __init__(
        self, vocabs: Vocabs, dim, dropout, enc_layers, dec_layers, bidirectional, 
        mean, std, dec_in_avg, m_bool, layer_norm, max_len):
        super().__init__(
            vocabs, dim, dropout, enc_layers, dec_layers, bidirectional, 
            mean, std, dec_in_avg, m_bool, layer_norm, max_len)
        self.null = nn.Linear(1,1)
    
    def init_qvals_network(self):
        pass
    
    def init_meta_network(self):
        pass
    
    def init_decoder_network(self):
        pass

    def forward(self, batch):
        x = torch.sum(batch['qvals_mask']*batch['qvals'],dim=1, keepdim=True)/(torch.sum(batch['qvals_mask'], dim=1, keepdim=True) + 1e-10)

        if(self.un_normalize):
            return self.do_unnorm(x)
        else:
            return x
        