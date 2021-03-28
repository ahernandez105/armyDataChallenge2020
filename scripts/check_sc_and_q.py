import os
import sys
import argparse
import pdb
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from util import preprocess

def init_argparser():
    args = argparse.ArgumentParser()
    args.add_argument('--sc', type=str, required=False, default='data/train/mission_sc_all')
    args.add_argument('--q', type=str, required=False, default='data/train/q_score_all')

    return args.parse_args()

def main(args):
    sc = open(args.sc, 'r').readlines()
    q = open(args.q, 'r').readlines()

    for i in range(len(sc)):
        sc_ = [float(x) for x in sc[i].strip().split()]
        q_ = [float(x) for x in q[i].strip().split()]
        for j in range(len(sc_)):
            if q_[j]==-1 or sc_[j]==-1:
                if q_[j] != sc_[j]:
                    print(i)
                    print(sc_)
                    print(q_)
                    # raise ValueError

if __name__=='__main__':
    main(init_argparser())