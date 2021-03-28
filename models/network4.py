import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import layer_norm

from models import EncodeAttentionQMnM1M2, FeedForward

class EncodeFeedForwardQMnMM(EncodeAttentionQMnM1M2):
    def __init__(
        self, vocabs, dim, dropout, enc_layer, dec_layers, bidirectional, 
        mean, std, dec_in_avg, m_bool, layer_norm, max_len):
        super().__init__(
            vocabs, dim, dropout, enc_layer, dec_layers, bidirectional, 
            mean, std, dec_in_avg, m_bool, layer_norm, max_len)
    
    def init_decoder_network(self):
        if self.bidirectional:
            hidden_size = self.dim * 2
        else:
            hidden_size = self.dim
        
        in_size = hidden_size*2*self.max_len + self.dim*2*5
        self.ff_all = FeedForward(
            in_features=in_size, out_features=in_size, layers=self.dec_layers, dropout=self.dropout,
            do_layer_norm=self.layer_norm)
        self.linear_dec1 = nn.Linear(in_features=in_size, out_features=int(in_size/3))
        self.linear_dec2 = nn.Linear(in_features=int(in_size/3), out_features=1)
    
    def forward(self, batch):
        enc_qvals, enc_mvals, enc_meta, enc_meta2 = self.do_encoding(batch)
        encs = [enc_qvals, enc_mvals, enc_meta, enc_meta2]
        for i in range(len(encs)):
            bsize, len_, dim = encs[i].shape
            encs[i] = encs[i].reshape(bsize, len_*dim)
        output = self.ff_all(torch.cat(encs, dim=1))
        output = self.dropout(F.relu(self.linear_dec1(output))) 
        output = self.linear_dec2(output)

        if(self.un_normalize):
            return self.do_unnorm(output)
        else:
            return output
