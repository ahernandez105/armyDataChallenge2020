import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import EncodeAttentionQMnM1M2M3

class EncodeAttentionQMnM1M2M3NoFF(EncodeAttentionQMnM1M2M3):
    def __init__(
        self, vocabs, dim, dropout, enc_layer, dec_layers, bidirectional, 
        mean, std, dec_in_avg, m_bool, layer_norm, max_len):
        super().__init__(
            vocabs, dim, dropout, enc_layer, dec_layers, bidirectional, 
            mean, std, dec_in_avg, m_bool, layer_norm, max_len)
        del self.ff_meta3
        del self.ff_meta2
        del self.ff_base
        del self.ff_hq
        del self.ff_rating
        del self.ff_region
        del self.ff_year
    
    def encode_meta(self, meta, col2idx):
        bsize, _ = meta.shape
        enc_base = self.dropout(self.emb_base(meta[:, col2idx['base']].type(torch.long)))
        enc_ratings = self.dropout(self.linear_rating(self.emb_ratings(meta[:, col2idx['rating']].type(torch.long))))
        enc_region = self.dropout(self.emb_region(meta[:, col2idx['region']].type(torch.long)))
        enc_hq = self.dropout(self.emb_hq(meta[:, col2idx['hq']].type(torch.long)))
        enc_year = self.dropout(self.linear_year(meta[:, [col2idx['year']]]))

        return torch.cat(
            [enc_year, enc_hq, enc_region, enc_base, enc_ratings], dim=1).reshape(bsize, 5, self.dim)
    
    def encode_meta2(self, meta, col2idx):
        bsize, n = meta.shape
        encs = []

        for i in range(len(self.meta2_cols)):
            encs.append(self.dropout(self.emb_meta2[self.meta2_cols[i]](meta[:, i])))
        
        return torch.cat(encs, dim=1).reshape(bsize, n, self.dim)
    
    def encode_meta3(self, meta3):
        bsize, n = meta3.shape
        encs = []

        for i in range(len(self.meta3_scols)):
            encs.append(self.dropout(self.emb_meta3[self.meta3_cols[i]](meta3[:, i].type(torch.long))))
        
        for i in range(len(self.meta3_scols), len(self.meta3_cols)):
            encs.append(self.dropout(self.emb_meta3[self.meta3_cols[i]](meta3[:, [i]])))
        
        return torch.cat(encs, dim=1).reshape(bsize, n, self.dim)