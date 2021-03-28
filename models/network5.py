import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import EncodeAttentionQMnM1M2, FeedForward

class EncodeAttentionQMnM1M2M3(EncodeAttentionQMnM1M2):
    def __init__(
        self, vocabs, dim, dropout, enc_layer, dec_layers, bidirectional, 
        mean, std, dec_in_avg, m_bool, layer_norm, max_len):
        super().__init__(
            vocabs, dim, dropout, enc_layer, dec_layers, bidirectional, 
            mean, std, dec_in_avg, m_bool, layer_norm, max_len)
        self.init_meta3_network()
        self.meta3_att = None
    
    def init_meta3_network(self):
        self.meta3_cols = [
            'org', 'instl', 'op_status', 'terr', 'type_desc', 'facility_book_value',
            'facility_replacement_value', 'linear_feet_total', 'acres_total', 'square_feet_total']
        self.meta3_scols = [
            'org', 'instl', 'op_status', 'terr', 'type_desc']
        self.meta3_fcols = [
            'facility_book_value', 'facility_replacement_value', 'linear_feet_total', 
            'acres_total', 'square_feet_total']
        col2vocab = {
            'org': self.vocabs.org, 'instl': self.vocabs.instl, 
            'op_status': self.vocabs.op_status, 'terr': self.vocabs.terr, 
            'type_desc': self.vocabs.type_desc}
        self.emb_meta3 = nn.ModuleDict({})
        self.ff_meta3 = nn.ModuleDict({})

        for i in range(len(self.meta3_scols)):
            self.emb_meta3[self.meta3_cols[i]] = nn.Embedding(
                num_embeddings=len(col2vocab[self.meta3_cols[i]]), embedding_dim=self.dim)
            self.ff_meta3[self.meta3_cols[i]] = FeedForward(
                self.dim, self.dim, self.enc_layers, self.dropout, self.layer_norm)
        
        for i in range(len(self.meta3_scols), len(self.meta3_cols)):
            self.emb_meta3[self.meta3_cols[i]] = nn.Linear(1, self.dim)
            self.ff_meta3[self.meta3_cols[i]] = FeedForward(
                self.dim, self.dim, self.enc_layers, self.dropout, self.layer_norm)
    
    def init_attention_network(self, dec_size, enc_size, enc_meta_size):
        super().init_attention_network(dec_size, enc_size, enc_meta_size)
        self.linear_att_meta3 = nn.Linear(in_features=enc_meta_size, out_features=enc_size, bias=False)
        self.linear_att_dec_meta3 = nn.Linear(in_features=enc_size, out_features=1, bias=True)
    
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
                input_size=hidden_size*4 + self.dim*(5), 
                hidden_size=hidden_size)
            self.init_attention_network(hidden_size, hidden_size, self.dim)

        else:
            self.lstm_dec = nn.LSTMCell(
                input_size=self.dim*(5) + hidden_size*2, 
                hidden_size=hidden_size)
            self.init_attention_network(self.dim, hidden_size, self.dim)
        
        # final output layer
        start_in = hidden_size*3 + self.dim*(3)
        self.linear_dec1 = nn.Linear(in_features=start_in, out_features=int(start_in/3))
        self.linear_dec2 = nn.Linear(in_features=int(start_in/3), out_features=1)
    
    def encode_meta3(self, meta3):
        bsize, n = meta3.shape
        encs = []

        for i in range(len(self.meta3_scols)):
            encs.append(
                self.ff_meta3[self.meta3_cols[i]](
                    self.dropout(self.emb_meta3[self.meta3_cols[i]](meta3[:, i].type(torch.long)))))
        
        for i in range(len(self.meta3_scols), len(self.meta3_cols)):
            encs.append(
                self.ff_meta3[self.meta3_cols[i]](
                    self.dropout(self.emb_meta3[self.meta3_cols[i]](meta3[:, [i]]))))
        
        return torch.cat(encs, dim=1).reshape(bsize, n, self.dim)

    def attention(self, ds, encoded, mask, mask2, dec_hidden):
        out = super().attention(ds, encoded, mask, mask2, dec_hidden)

        if out is not None:
            return out
        
        elif ds=='meta3':
            s = self.linear_att_dec_q(dec_hidden)
            h = self.linear_att_meta3(encoded)
            scores = self.linear_att_dec_meta3(torch.tanh(s.reshape(s.shape[0], 1, s.shape[1]) + h))
            if self.m_bool:
                scores.masked_fill_((1-mask.unsqueeze(-1)).type(torch.bool), -1e8)
            else:
                scores.masked_fill_((1-mask.unsqueeze(-1)).byte(), -1e8)
            weights = F.softmax(scores, dim=1)
            weights = weights * (mask2.unsqueeze(-1))
            out = torch.sum(weights * encoded, dim=1, keepdim=False)
            self.meta3_att = weights

            return out
        else:
            return None
    
    def do_encoding(self, batch):
        enc_qvals, enc_mvals, enc_meta, enc_meta2 = super().do_encoding(batch)
        enc_meta3 = self.encode_meta3(batch['meta3'])

        return enc_qvals, enc_mvals, enc_meta, enc_meta2, enc_meta3
    
    def do_attention(self, batch, dec_in_map, encoded_map):
        ctx_qvals, ctx_mvals, ctx_meta, ctx_meta2 = super().do_attention(batch, dec_in_map, encoded_map)
        ctx_meta3 = self.attention(
            'meta3', encoded_map['meta3'], batch['meta3_mask'], batch['meta3_mask2'], dec_in_map['qvals'])
        
        return ctx_qvals, ctx_mvals, ctx_meta, ctx_meta2, ctx_meta3
    
    def forward(self, batch):
        enc_qvals, enc_mvals, enc_meta, enc_meta2, enc_meta3 = self.do_encoding(batch)
        encoded_map = {'qvals': enc_qvals, 'mvals': enc_mvals, 'meta': enc_meta, 'meta2': enc_meta2, 'meta3': enc_meta3}
        dec_in_qvals, dec_in_mvals, hidden_init = self.do_encoding_dec_inputs(batch, encoded_map)
        dec_in_map = {'qvals': dec_in_qvals, 'mvals': dec_in_mvals}
        ctx_qvals, ctx_mvals, ctx_meta, ctx_meta2, ctx_meta3 = self.do_attention(batch, dec_in_map, encoded_map)

        # do decoding
        output, _ = self.lstm_dec(
            torch.cat([dec_in_qvals, dec_in_mvals, ctx_qvals, ctx_mvals, ctx_meta, ctx_meta2, ctx_meta3], dim=1), 
            (hidden_init, hidden_init)) 
        output = self.dropout(
            F.relu(self.linear_dec1(torch.cat([output, ctx_qvals, ctx_mvals, ctx_meta, ctx_meta2, ctx_meta3], dim=1)))) 
        output = self.linear_dec2(output)

        if(self.un_normalize):
            return self.do_unnorm(output)
        else:
            return output

        
