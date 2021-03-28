import os
import json
from typing import Dict
import pdb

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
 
class Vocabulary:
    def __init__(self, path):
        self.map = dict()
        self.imap = dict()
        self.path = path

        with open(self.path, 'r') as file:
            _json = json.load(file)
        
        for key in _json:
            self.map[key] = int(_json[key][0])
        
        for item in self.map.items():
            self.imap[item[1]] = item[0]
    
    def word2idx(self, word):
        return self.map.get(word, self.map['<unk>'])

    def idx2word(self, idx):
        return self.imap.get(idx, '<unk>')
    
    def __len__(self):
        return len(self.map)

class Vocabs:
    def __init__(self, paths_dict):
        self.ratings = Vocabulary(paths_dict['ratings'])
        self.hq = Vocabulary(paths_dict['hq'])
        self.base = Vocabulary(paths_dict['base'])
        self.region = Vocabulary(paths_dict['region'])
        if 'faceno' in paths_dict:
            self.faceno = Vocabulary(paths_dict['faceno'])
            self.rpa_op = Vocabulary(paths_dict['rpa_op'])
            self.rpa_type = Vocabulary(paths_dict['rpa_type'])
            self.rpa_inter = Vocabulary(paths_dict['rpa_inter'])
            self.construct = Vocabulary(paths_dict['construct'])
        if 'org' in paths_dict:
            self.org = Vocabulary(paths_dict['org'])
            self.instl = Vocabulary(paths_dict['instl'])
            self.op_status = Vocabulary(paths_dict['op_status'])
            self.terr = Vocabulary(paths_dict['terr'])
            self.type_desc = Vocabulary(paths_dict['type_desc'])

class DatasetAbstract(Dataset):
    def __init__(self, path, in_mem):
        self.path = path
        self.data = None

        if in_mem:
            with open(self.path, 'r') as file:
                self.data = []
                for i, line in enumerate(file):
                    self.data.append(line.strip())

    def _get_example(self, idx):
        if self.data is not None:
            return self.data[idx]
        else:
            with open(os.path.join(self.path, str(idx)), 'r') as file:
                return file.readline()
    
    def __len__(self):
        if self.data is not None:
            return len(self.data)
        else:
            return len(os.listdir(self.path))
    
    def __getitem__(self, idx):
        raise NotImplementedError

    def collate_fn(self, batch):
        raise NotImplementedError

class Skus(DatasetAbstract):
    def __init__(self, path, in_mem, device):
        super().__init__(path, in_mem)
        self.device = device
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        return {
            'skus': torch.tensor([float(self._get_example(idx))], dtype=torch.float, device=self.device)}
    
    def collate_fn(self, batch):
        return {'skus': torch.stack([batch[i]['skus'] for i in range(len(batch))])}


class  TimeSeriesEncDec(DatasetAbstract):
    QVAL_MEANS = [84.931261, 78.181053, 85.209234, 85.277987, 84.925561, 85.434429]
    QVAL_STDS = [15.437801, 26.168610, 15.320776, 15.480150, 15.429693, 14.8311857]
    MVAL_MEANS = [47.616814, 47.971822, 66.608804, 70.530634, 73.024413, 83.100261]
    MVAL_STDS = [41.541946, 39.1033804, 32.899082, 29.187656, 26.780303, 17.311207]
    QIC_MEANS = [139339.86916513598, 130775.4249951094, 157430.205537088, 162742.9779345311, 153874.15887430054, 157136.1032771572]
    QIC_STDS = [3732284.628655126, 1828632.0761620896, 6051922.032779889, 8078182.803862358, 2522131.0214385106, 2585981.21484381]
    PAD = -5
    def __init__(
        self, path, device, in_mem, max_len, filter_nan, key, means, stds, 
        dec_in_avg, rating_path=None, vocab: Vocabulary=None):
        super().__init__(path, in_mem)
        self.device = device
        self.key = key
        self.keys = {
            'key': key, 'mask': key+'_'+'mask', 'mask2': key+'_'+'mask2', 
            'dec_in': key+'_'+'dec_in', 'ratings': key+"_"+"ratings"}
        self.max_len = max_len
        self.means = means
        self.stds = stds
        self.filter_nan = filter_nan
        self.dec_in_avg = dec_in_avg
        if self.dec_in_avg:
            self.val_len = self.max_len
        else:
            self.val_len = self.max_len -1

        self.vocab = vocab
        self.ratings_path = rating_path
        self.data_ratings = None
        if in_mem and self.ratings_path is not None:
            with open(self.ratings_path, 'r') as file:
                self.data_ratings = []
                for i, line in enumerate(file):
                    self.data_ratings.append(line.strip())
    
    def _get_example(self, idx):
        out = []

        if self.data is not None:
            out.append(self.data[idx])
        else:
            with open(os.path.join(self.path, str(idx)), 'r') as file:
                out.append(file.readline())
        
        if self.ratings_path is not None:
            if self.data_ratings is not None:
                out.append(self.data_ratings[idx])
            else:
                with open(os.path.join(self.data_ratings, str(idx)), 'r') as file:
                    out.append(file.readline())
        
        return out
    
    def normalize(self, x, mean, std):
        if x>=0:
            return (x - mean)/std
        else:
            return TimeSeriesEncDec.PAD
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        data = []
        ex = self._get_example(idx)
        ex[0] = [float(x) for x in ex[0].split()[0:self.max_len]]
        dm = [1 if x>=0 else 0 for x in ex[0]]
        data.append([self.normalize(ex[0][i], self.means[i], self.stds[i]) for i in range(len(ex[0]))])

        if len(ex)>1:
            data.append([self.vocab.word2idx(x) for x in ex[1].split()[0:self.max_len+1]])
            r = [self.vocab.word2idx('<pad>') for i in range(len(data[1]))]
        
        if sum(dm)==1 or sum(dm)==0:
            out = {
                    self.keys['key']: torch.tensor([TimeSeriesEncDec.PAD for i in range(self.val_len)], dtype=torch.float, device=self.device),
                    self.keys['mask']: torch.tensor([0 for i in range(self.val_len)], dtype=torch.float, device=self.device),
                    self.keys['mask2']: torch.tensor([0 for i in range(self.val_len)], dtype=torch.float, device=self.device)}
            
            if sum(dm)==1:
                dec_in = [data[0][i] for i in range(len(dm)) if dm[i]==1]
                out[self.keys['dec_in']] = torch.tensor(dec_in, dtype=torch.float, device=self.device)
                if len(data)>1:
                    if self.dec_in_avg:
                        r[-1] = data[1][-1]
                    else:
                        r[-2] = [data[1][i] for i in range(len(dm)) if dm[i]==1][0]
                        r[-1] = data[1][-1]
                    out[self.keys['ratings']] = torch.tensor(r, dtype=torch.float, device=self.device)
            else:
                out[self.keys['dec_in']] = torch.tensor([TimeSeriesEncDec.PAD], dtype=torch.float, device=self.device)
                if len(data)>1:
                    out[self.keys['ratings']] = torch.tensor(r, dtype=torch.float, device=self.device)
        else:
            dec_in = None
            enc_in = [TimeSeriesEncDec.PAD for i in range(self.val_len)]
            mask = [0 for i in range(self.val_len)]

            if self.filter_nan:
                vals = [data[0][i] for i in range(len(dm)) if dm[i]==1]
                if len(data)>1:
                    vals_r = [data[1][i] for i in range(len(dm)) if dm[i]==1]
        
                if self.dec_in_avg:
                    enc_in[0: len(vals)] = vals
                    mask[0: len(vals)] = [1 for i in range(len(vals))]
                    dec_in = sum(vals)/len(vals)

                    if len(data)>1:
                        r[0: len(vals_r)] = vals_r
                        r[-1] = data[1][-1]
                else:
                    enc_in[0: len(vals)-1] = vals[0:-1]
                    mask[0: len(vals)-1] = [1 for i in range(len(vals)-1)]
                    dec_in = vals[-1]
                    if len(data)>1:
                        r[0: len(vals_r)-1] = vals_r[0:-1]
                        r[-2] = vals_r[-1]
                        r[-1] = data[1][-1]
            else:
                if len(data)>1:
                    r = data[1]
                if self.dec_in_avg:
                    enc_in = data[0]
                    dec_in = sum([data[0][i] for i in range(self.val_len) if dm[i]==1])/sum(dm)
                    mask = dm
                else:
                    enc_in = data[0][0:-1]
                    dec_in = data[0][-1]
                    mask = dm[0:-1]

            out = {
                    self.keys['key']: torch.tensor(enc_in, dtype=torch.float, device=self.device),
                    self.keys['mask']: torch.tensor(mask, dtype=torch.float, device=self.device),
                    self.keys['mask2']: torch.tensor([1 for i in range(self.val_len)], dtype=torch.float, device=self.device),
                    self.keys['dec_in']: torch.tensor([dec_in],dtype=torch.float, device=self.device)}
            if len(data)>1:
                out[self.keys['ratings']] = torch.tensor(r, dtype=torch.float, device=self.device)
        
        return out
    
    def collate_fn(self, batch):
        bsize = range(len(batch))
        out = {
            self.keys['key']: torch.stack([batch[i][self.keys['key']] for i in bsize]),
            self.keys['mask']: torch.stack([batch[i][self.keys['mask']] for i in bsize]),
            self.keys['mask2']: torch.stack([batch[i][self.keys['mask2']] for i in bsize]),
            self.keys['dec_in']: torch.stack([batch[i][self.keys['dec_in']] for i in bsize])}
        if self.ratings_path is not None:
            out[self.keys['ratings']] = torch.stack([batch[i][self.keys['ratings']] for i in bsize])
        
        return out

class Meta(DatasetAbstract):
    def __init__(self, path, vocabs: Vocabs, device, in_mem):
        super().__init__(path, in_mem)
        self.device = device
        self.vocab = vocabs
        self.col2idx = {'year': 0, 'hq': 1, 'region': 2, 'base': 3, 'rating': 4}
        self.keys = ['meta']
        self.mean = 1985.983079
        self.std = 32.744022
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        data = self._get_example(idx).split()[:-1]
        data[self.col2idx['year']] = (float(data[self.col2idx['year']]) - self.mean)/self.std
        data[self.col2idx['hq']] = self.vocab.hq.word2idx(data[self.col2idx['hq']])
        data[self.col2idx['region']] = self.vocab.region.word2idx(data[self.col2idx['region']])
        data[self.col2idx['base']] = self.vocab.base.word2idx(data[self.col2idx['base']])
        
        return {'meta': torch.tensor(data, dtype=torch.float, device=self.device)}
    

    def collate_fn(self, batch):
        return {
            'meta': torch.stack([batch[i]['meta'] for i in range(len(batch))]),
            'col2idx': self.col2idx}

class Meta2(DatasetAbstract):
    def __init__(self, path, vocabs, device, in_mem):
        super().__init__(path, in_mem)
        self.device = device
        self.vocabs = vocabs
        self.cols = ['faceno', 'rpa_op', 'rpa_type', 'rpa_inter', 'construct']
        self.col2idx = {'faceno': 0, 'rpa_op': 1, 'rpa_type': 2, 'rpa_inter': 3, 'construct': 4}
        self.col2vocab = {
            'faceno': self.vocabs.faceno, 'rpa_op': self.vocabs.rpa_op,
            'rpa_type': self.vocabs.rpa_type, 'rpa_inter': self.vocabs.rpa_inter,
            'construct': self.vocabs.construct}
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        data = self._get_example(idx).split()
        x = data.copy()
        for i in range(len(self.cols)):
            data[i] = self.col2vocab[self.cols[i]].word2idx(data[i])
        
        return {'meta2': torch.tensor(data, dtype=torch.long, device=self.device)}
    
    def collate_fn(self, batch):
        return {
            'meta2': torch.stack([batch[i]['meta2'] for i in range(len(batch))]),
            'col2idx_2': self.col2idx}

class Meta3(DatasetAbstract):
    def __init__(self, path, vocabs: Vocabs, device, in_mem):
        super().__init__(path, in_mem)
        self.device = device
        self.vocabs = vocabs
        self.__cols__ = [
            'accountable_sub_org_name', 'instl_name', 'op_status', 'st_terr_name', 'type_desc',
            'facility_book_value','facility_replacement_value', 'linear_feet_total', 'acres_total', 
            'square_feet_total']
        self.cols = [
            'org', 'instl', 'op_status', 'terr', 'type_desc',
            'facility_book_value','facility_replacement_value', 
            'linear_feet_total', 'acres_total', 'square_feet_total']
        self.str_cols = ['org', 'instl', 'op_status', 'terr', 'type_desc']
        self.col2vocab = {
            'org': self.vocabs.org, 'instl': self.vocabs.instl, 'op_status': self.vocabs.op_status,
            'terr': self.vocabs.terr, 'type_desc': self.vocabs.type_desc}
        self.float_cols = [
            'facility_book_value','facility_replacement_value', 'linear_feet_total', 
            'acres_total', 'square_feet_total']
        self.means = [375970.98597884754, 1255528.887779204, 11390.932598472034, 708.1251548946716, 6728.261827028724]
        self.stds = [3484840.6668070466, 10287510.37342274, 668974.608604989, 15751.128298827327, 22851.752618879607]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        data = self._get_example(idx).split()
        out_data = []
        mask = []

        for i in range(len(self.str_cols)):
            out_data.append(self.col2vocab[self.str_cols[i]].word2idx(data[i]))
            if data[i]=='nan':
                mask.append(0) 
            else:
                mask.append(1)
        
        for i in range(len(self.str_cols), len(self.cols)):
            idx = i - len(self.str_cols)
            if data[i]=='nan':
                out_data.append(-5)
                mask.append(0)
            else:
                out_data.append((float(data[i])-self.means[idx])/self.stds[idx])
                mask.append(1)
        
        if sum(mask)==0:
            mask2 = [0 for i in range(len(self.cols))]
        else:
            mask2 = [1 for i in range(len(self.cols))]

        return {
            'meta3': torch.tensor(out_data, dtype=torch.float, device=self.device),
            'meta3_mask': torch.tensor(mask, dtype=torch.long, device=self.device),
            'meta3_mask2': torch.tensor(mask2, dtype=torch.float, device=self.device)}
        
    def collate_fn(self, batch):
        return {
            'meta3': torch.stack([batch[i]['meta3'] for i in range(len(batch))]),
            'meta3_mask': torch.stack([batch[i]['meta3_mask'] for i in range(len(batch))]),
            'meta3_mask2': torch.stack([batch[i]['meta3_mask2'] for i in range(len(batch))])}

class Targets(DatasetAbstract):
    def __init__(self, path, device, in_mem):
        super().__init__(path, in_mem)
        self.keys = ['targets']
        self.device = device
        self.mean = sum([85.209234, 85.277987, 84.925561, 85.434429])/4
        self.std = sum([15.320776, 15.480150, 15.429693, 14.8311857])/4
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        data = float(self._get_example(idx))
        
        return {
            'targets': torch.tensor([(data-self.mean)/self.std], dtype=torch.float, device=self.device),
            'targets_': torch.tensor([data], dtype=torch.float, device=self.device)}
    
    def collate_fn(self, batch):
        return {
            'targets': torch.stack([batch[i]['targets'] for i in range(len(batch))]),
            'targets_': torch.stack([batch[i]['targets_'] for i in range(len(batch))])}

class Multimodal(Dataset):
    def __init__(self, paths_dict, vocabs: Vocabs, device, in_mem, max_len, filter_nan, dec_in_avg):
        self.vocab = vocabs
        self.datasets = []
        self.meta = Meta(paths_dict['meta'], self.vocab, device, in_mem)
        self.datasets.append(self.meta)
        self.skus = Skus(paths_dict['skus'], in_mem, device)
        self.datasets.append(self.skus)
        if os.path.exists(paths_dict['targets']):
            self.targets = Targets(paths_dict['targets'], device, in_mem)
            self.datasets.append(self.targets)
        self.qvals = TimeSeriesEncDec(
            paths_dict['qvals'], device, in_mem, max_len, filter_nan, 'qvals',
            TimeSeriesEncDec.QVAL_MEANS, TimeSeriesEncDec.QVAL_STDS, dec_in_avg,
            paths_dict['ratings'], self.vocab.ratings)
        self.datasets.append(self.qvals)

        if 'meta2' in paths_dict:
            self.meta2 = Meta2(paths_dict['meta2'], vocabs, device, in_mem)
            self.datasets.append(self.meta2)
        
        if 'mvals' in paths_dict:
            self.mvals = TimeSeriesEncDec(
                paths_dict['mvals'], device, in_mem, max_len, filter_nan, 'mvals', 
                TimeSeriesEncDec.MVAL_MEANS, TimeSeriesEncDec.MVAL_STDS, dec_in_avg) 
            self.datasets.append(self.mvals)
        
        if 'qic' in paths_dict:
            self.qic = TimeSeriesEncDec(
                paths_dict['qic'], device, in_mem, max_len, filter_nan, 'qic', 
                TimeSeriesEncDec.QIC_MEANS, TimeSeriesEncDec.QIC_STDS, dec_in_avg)
            self.datasets.append(self.qic)
        
        if 'meta3' in paths_dict:
            self.meta3 = Meta3(paths_dict['meta3'], self.vocab, device, in_mem)
            self.datasets.append(self.meta3)

    def __len__(self):
        return len(self.qvals)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        out = {}
        for ds in self.datasets:
            out.update(ds[idx])

        return out
    
    def collate_fn(self, batch):
        out = {}
        for ds in self.datasets:
            out.update(ds.collate_fn(batch))

        return out

class Multimodal2(Multimodal):
    def __init__(self, paths_dict, vocabs: Vocabs, device, in_mem, max_len, filter_nan, dec_in_avg):
        super().__init__(paths_dict, vocabs, device, in_mem, max_len, filter_nan, dec_in_avg)
        self.include_enc = ['qvals']
        if 'qic' in paths_dict:
            self.include_enc.append('qic')
        if 'mvals' in paths_dict:
            self.include_enc.append('mvals')
    
    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        masks = [out[ds+'_mask'] for ds in self.include_enc]
        enc_mask = [0 for i in range(self.qvals.val_len)]

        for i in range(self.qvals.val_len):
            ttl = sum([masks[ds][i].item() for ds in range(len(self.include_enc))])
            if ttl>=1:
                enc_mask[i] = 1
            else:
                enc_mask[i] = 0
        
        if sum(enc_mask)==0:
            enc_mask2 = [0 for i in range(self.qvals.val_len)]
        else:
            enc_mask2 = [1 for i in range(self.qvals.val_len)]
        
        out['enc_mask'] = torch.tensor(enc_mask, dtype=torch.float, device=self.qvals.device)
        out['enc_mask2'] = torch.tensor(enc_mask2, dtype=torch.float, device=self.qvals.device)
    
        return out
    
    def collate_fn(self, batch):
        out = super().collate_fn(batch)
        out['enc_mask'] = torch.stack([batch[i]['enc_mask'] for i in range(len(batch))])
        out['enc_mask2'] = torch.stack([batch[i]['enc_mask2'] for i in range(len(batch))])

        return out
        

        







