import os
import sys
import argparse
import pdb
from collections import Counter
import operator
import json
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from util import preprocess

def init_argparser():
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, required=False, default='data/train/meta')
    args.add_argument('--col', type=int, required=False, default=1)
    args.add_argument('--dest', type=str, required=False, default='data/vocabs/head_quarter')
    args.add_argument('--exclude_sym', type=str, required=False, default='')
    args.add_argument('--min', type=int, default=1, required=False)

    return args.parse_args()

def main(args):
    tokens = []

    with open(args.data, 'r') as file:
        for i,line in enumerate(file):
            token = line.strip().split()[args.col]
            if token != args.exclude_sym:
                tokens.append(token)
    
    counts = dict(Counter(tokens))
    sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    
    with open(args.dest, 'w') as file:
        file.write('{\n')
        file.write('"<unk>":["0","0"],\n')
        file.write('"<pad>":["1","0"],\n')
        lines = []
        for i in range(len(sorted_counts)):
            if sorted_counts[i][1] < args.min:
                break
            else:
                line = f'"{sorted_counts[i][0]}": ["{i+2}","{sorted_counts[i][1]}"]'
                lines.append(line)
        
        file.write(",\n".join(lines))
        file.write('\n}')



if __name__=='__main__':
    args = init_argparser()
    main(args)
    with open(args.dest, 'r') as file:
        print(json.load(file))
