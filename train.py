import argparse
import enum
import os
import logging
import re
import sys
from datetime import datetime
from math import ceil
from pathlib import Path
from progress.bar import IncrementalBar

import torch
from torch import storage
from torch.utils.data import DataLoader
import torch.nn as nn

import models
from models import EncodeAttentionQnM1, EncodeAttentionQMnM1M2, EncodeAttentionQMQnMM
from models import EncodeFeedForwardQMnMM, Average, EncodeAttentionQMnM1M2M3
from models import EncodeAttentionQMnM1M2M3NoFF, EncodeSmallAttentionQMnM1M2M3NoFF
import datasets
from datasets import Vocabs, Multimodal, Multimodal2

def init_argparser():
    args = argparse.ArgumentParser()
    args.add_argument('--epochs', type=int, required=False, default=150)
    args.add_argument('--train', type=str, required=False, default='data/train')
    args.add_argument('--valid', type=str, required=False, default='data/valid')
    args.add_argument('--test', type=str, required=False, default='data/test')
    args.add_argument('--vocab', type=str, required=False, default='data/vocabs')
    args.add_argument('--bsize', type=int, required=False, default=256)
    args.add_argument('--device', type=str, required=False, default='cpu')
    args.add_argument('--lr', type=float, required=False, default=0.004)
    args.add_argument('--patience', type=int, required=False, default=10)
    args.add_argument('--dim', type=int, required=False, default=80)
    args.add_argument('--dropout', type=float, required=False, default=0.1)
    args.add_argument('--enc_layers', type=int, required=False, default=2)
    args.add_argument('--dec_layers',type=int, required=False, default=1)
    args.add_argument('--log_interval', type=int, required=False, default=20)
    args.add_argument('--clip', type=float, required=False, default=1)
    args.add_argument('--decay_lr', required=False, action='store_true', default=True)
    args.add_argument('--ram_data', required=False, action='store_true', default=True)
    args.add_argument('--max_len', required=False, type=int, default=5)
    args.add_argument('--include_meta2', required=False, default=True, action='store_true')
    args.add_argument('--ds', required=False, type=str, default='Multimodal')
    args.add_argument('--bidirectional', action='store_true', required=False)
    args.add_argument('--layernorm', action='store_true', required=False)
    args.add_argument('--loss', required=False, default='MSELoss', type=str)
    args.add_argument('--include_qic', required=False, action='store_true')
    args.add_argument('--include_meta3', required=False, action='store_true')
    args.add_argument('--m_bool', required=False, action='store_true')
    args.add_argument('--filter_nan', required=False, action='store_true')
    args.add_argument('--dec_in_avg', required=False, action='store_true')
    args.add_argument('--net', required=False, type=str, default='EncodeAttentionQnM1')
    args.add_argument('--save', type=str, required=True, default='')
    args.add_argument('--log', type=str, required=True, default='')
    args.add_argument('--checkpoint', type=str, required=False, default='')
    args.add_argument('--submit', type=str, required=False, default=None)

    args = args.parse_args()

    args.save = os.path.join(os.getcwd(), args.save)
    args.log = os.path.join(os.getcwd(), args.log)
    args.train = os.path.join(os.getcwd(), args.train)
    args.valid = os.path.join(os.getcwd(), args.valid)
    args.test = os.path.join(os.getcwd(), args.test)
    args.vocab = os.path.join(os.getcwd(), args.vocab)

    if not Path(args.log).parent.exists():
        os.mkdir(str(Path(args.log).parent))
    
    if not Path(args.save).parent.exists():
        os.mkdir(str(Path(args.save).parent))

    return args

def init_logger(args):
    if os.path.exists(args.log):
        os.remove(args.log)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(args.log))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger

def print_args(args, logger):
    _args = str(args.__repr__()).replace("Namespace","").replace("(","").replace(")","").split(',')

    logger.debug("Setup:")
    for arg in _args:
        key, value=arg.split('=')
        logger.debug(key.strip() + ": " + value.strip())
    logger.debug("")

def get_vocab_dict(path, args):
    out = {
        'ratings': os.path.join(path, 'rating'), 'hq': os.path.join(path, 'head_quarter'),
        'base': os.path.join(path, 'base'), 'region': os.path.join(path, 'region')
    }

    if args.include_meta2:
        out['faceno'] = os.path.join(path, 'faceno')
        out['rpa_op'] = os.path.join(path, 'rpa_op')
        out['rpa_type'] = os.path.join(path, 'rpa_type')
        out['rpa_inter'] = os.path.join(path, 'rpa_inter')
        out['construct'] = os.path.join(path, 'construct')
    if args.include_meta3:
        out['org'] = os.path.join(path, 'org')
        out['instl'] = os.path.join(path, 'instl')
        out['op_status'] = os.path.join(path, 'op_status')
        out['terr'] = os.path.join(path, 'terr')
        out['type_desc'] = os.path.join(path, 'type_desc')

    return out 

def get_data_dict(path, args):
    out = {
        'qvals': os.path.join(path, 'q_score_all'), 'ratings': os.path.join(path, 'ratings_all'),
        'meta': os.path.join(path, 'meta'), 'targets': os.path.join(path, 'targets'),
        'mvals': os.path.join(path, 'mission_sc_all'), 'skus': os.path.join(path, 'skus')}

    if args.include_meta2:
        out['meta2'] = os.path.join(path, 'meta2')
    if args.include_qic:
        out['qic'] = os.path.join(path, 'qic')
    if args.include_meta3:
        out['meta3'] = os.path.join(path, 'meta3')

    return out

def init_data(args):
    vocabs = Vocabs(get_vocab_dict(args.vocab, args))
    train = getattr(datasets, args.ds)(
        get_data_dict(args.train, args), vocabs, args.device, args.ram_data, 
        args.max_len, args.filter_nan, args.dec_in_avg)
    valid = getattr(datasets, args.ds)(
        get_data_dict(args.valid, args), vocabs, args.device, args.ram_data, 
        args.max_len, args.filter_nan, args.dec_in_avg)
    test = getattr(datasets, args.ds)(
        get_data_dict(args.test, args), vocabs, args.device, args.ram_data, 
        args.max_len, args.filter_nan, args.dec_in_avg)

    if args.device=='cuda':
        num_workers=0
    else:
        num_workers=0
    
    return (
        vocabs, 
        DataLoader(train, batch_size=args.bsize, num_workers=num_workers, shuffle=True, collate_fn=train.collate_fn),
        DataLoader(valid, batch_size=args.bsize, num_workers=num_workers, shuffle=False, collate_fn=valid.collate_fn),
        DataLoader(test, batch_size=args.bsize, num_workers=num_workers, shuffle=False, collate_fn=test.collate_fn))

def get_model(args, vocabs, mean, std):
    return getattr(models, args.net)(
        vocabs, args.dim, args.dropout, args.enc_layers, args.dec_layers, args.bidirectional, 
        mean, std, args.dec_in_avg, args.m_bool, args.layernorm, args.max_len)

def training(args, model: torch.nn.Module, train: DataLoader, valid: DataLoader, logger):
    criterion = getattr(torch.nn, args.loss)(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss, best_val_e, head, vhead = float('Inf'), float('nan'), '-'*115, '+'*115

    epoch=0
    for epoch in range(1,args.epochs+1):
        model.train()
        model.un_normalize = False
        loss, loss_, epoch_time = 0, 0, 0
        for i, batch in enumerate(train):
            i+=1
            batch_loss, batch_loss_, time = train_batch(batch, model, optimizer, criterion, args.clip)
            loss+=batch_loss
            loss_+=batch_loss_
            epoch_time+=time
             
            if (i % args.log_interval)==0:
                logger.debug(
                    "{h}\ntrain | epoch = {e:03d} | batch {b:04d}/{s:04d} ({t:.4f} secs./batch) | lr = {lr:.9f} | loss = {l:,.4f} | loss_ = {l_:,.2f}\n{h2}".format(
                    h=head, e=epoch, b=i, s=len(train), lr= args.lr, t=epoch_time/(i), 
                    l=loss/(i*args.bsize), l_=loss_/(i*args.bsize), h2=head))

        logger.debug("epoch {e:03d} train time = {t:.4f} mins".format(e=epoch, t=epoch_time/60))

        # do validation
        val_loss, val_time = evaluation(model, valid, args)
        temp_best_val_loss = val_loss if val_loss < best_val_loss else best_val_loss
        temp_best_val_e = epoch if val_loss < best_val_loss else best_val_e
        logger.debug(
            "{h1}\nvalid | epoch = {e:03d} | patience = {p} | total time {t:.2f} secs. | cur loss_ = {l:.2f} | best loss_ = {bl:.2f} (at {be:03d})\n{h2}".format(
            h1=vhead, e=epoch, p=args.patience, t=val_time, l=val_loss, 
            bl=temp_best_val_loss, be=temp_best_val_e, h2=vhead))
        
        # run patience and save checks
        if val_loss<best_val_loss:
            logger.debug("Saving the best model...")
            best_val_loss, best_val_e = val_loss, epoch
            torch.save(model.state_dict(), args.save)
        else:
            if args.decay_lr:
                logger.debug("Decreasing learning rate and patience...")
                args.lr/=4
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
            else:
                logger.debug('Decreasing patience...')

            args.patience-=1
            if(args.patience==0):
                break
    
    logger.debug("Saving model at epoch {e}".format(e=epoch)) # type: ignore
    torch.save(model.state_dict(), args.save+'.last')
    return False

def evaluation(model: torch.nn.Module, ds: DataLoader, args):
    criterion = torch.nn.MSELoss(reduction='sum')
    loss, time = 0, 0
    model.eval()
    model.un_normalize = True

    for i, batch in enumerate(ds):
        t = datetime.now()
        loss += criterion(model(batch), batch['targets_']).item()
        time += (datetime.now() - t).total_seconds()
    
    return loss/len(ds.dataset.targets), time


def train_batch(batch, model: torch.nn.Module, optimizer, criterion, clip):
    start = datetime.now()
    criterion2 = torch.nn.MSELoss(reduction='sum')

    optimizer.zero_grad()
    preds = model(batch)
    loss = criterion(preds, batch['targets'])
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    loss_ = criterion2(model.do_unnorm(preds), batch['targets_']).item()


    return loss.item(), loss_, (datetime.now() - start).total_seconds()

def do_submit(model: EncodeSmallAttentionQMnM1M2M3NoFF, ds, args):
    bar = IncrementalBar("Countdown", suffix="%(index)d/%(max)d - %(eta)ds", max=len(ds))
    model.eval()
    model.un_normalize = True
    lines = []
    att_lines = []

    for i, batch in enumerate(ds):
        preds = model(batch)
        floor = torch.zeros(preds.shape, dtype=torch.float, device=args.device)
        ceiling = torch.zeros(preds.shape, dtype=torch.float, device=args.device)
        ceiling[:,:] = 100
        preds = torch.max(floor, preds)
        preds = torch.min(ceiling, preds)
        for i in range(preds.shape[0]):
            lines.append('{sku},{p}'.format(sku=int(batch['skus'][i].item()), p=preds[i].item()))
            att_line = "{sku}".format(sku=int(batch['skus'][i].item()))+"\t"
            att_line += " ".join(["{x:.3f}".format(x=model.q_att[i,j].item()) for j in range(model.q_att.shape[1])])+"\t"
            att_line += " ".join(["{x:.3f}".format(x=model.meta_att[i,j].item()) for j in range(model.meta_att.shape[1])])+"\t"
            att_line += " ".join(["{x:.3f}".format(x=model.meta2_att[i,j].item()) for j in range(model.meta2_att.shape[1])])+"\t"
            att_line += " ".join(["{x:.3f}".format(x=model.meta3_att[i,j].item()) for j in range(model.meta3_att.shape[1])])
            att_lines.append(att_line)
        
        bar.next()
        break
    
    with open(args.submit, 'w') as file:
        file.write('\n'.join(lines))
    
    with open(str(Path(args.submit).parent.joinpath('att')), 'w') as file:
        file.write('\n'.join(att_lines))

def main(args, logger):
    vocabs, train, valid, test = init_data(args)
    model = get_model(args, vocabs, train.dataset.targets.mean, train.dataset.targets.std)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device(args.device)))
        logger.debug("Loaded model...")

    if args.device=='cuda':
        model.cuda()
    
    if args.submit is not None:
        do_submit(model, test, args)
        sys.exit()
    
    logger.debug(model.__repr__())
    logger.debug("Params: " + "{:,}".format(model.n_params()) + "\n")

    while training(args, model, train, valid, logger):
        pass

    logger.debug("Finished training...")
    logger.debug("Evaluating test set using best valid model...")
    model.load_state_dict(torch.load(args.save))
    loss, _ = evaluation(model, test, args)
    logger.debug("test loss_ = {:,.2f}".format(loss))

    logger.debug("Evaluating test set using last saved model...")
    model.load_state_dict(torch.load(args.save+'.last'))
    loss, _ = evaluation(model, test, args)
    logger.debug("test loss_ = {:,.2f}".format(loss))

if __name__=='__main__':
    args = init_argparser()
    logger = init_logger(args)

    if torch.cuda.is_available():
        if args.device=='cpu':
            logger.debug('Cuda is available. Setting device to cuda...')
            args.device='cuda'
    else:
        if args.device=='cuda':
            logger.debug('Cuda is not available. Setting device to cpu...')
            args.device='cpu'

    print_args(args, logger)
    main(args, logger)