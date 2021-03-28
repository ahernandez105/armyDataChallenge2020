import os
import sys
import argparse
import pdb
sys.path.append(os.getcwd())

import pandas as pd

from util import preprocess

def init_argparser():
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, required=False, default='data/csv/Q Score Summary.txt')
    args.add_argument('--skus', type=str, required=False, default='data/train/skus')
    args.add_argument('--dest', type=str, required=False, default='data/train/targets')

    return args.parse_args()

def read_data(path):
    return pd.read_csv(path, header=0, usecols=['Asset_UID', 'FY19 Asset Level Data_Qual_Score'])

def main(args):
    targets = preprocess.sort_df_on_skus(read_data(args.data), preprocess.get_skus_df(args.skus))['FY19 Asset Level Data_Qual_Score'].values
    print('MEAN')
    print(targets.mean())
    print('STD')
    print(targets.std())
    targets = [str(x) for x in targets]

    with open(args.dest, 'w') as file:
        file.write("\n".join(targets))
    
if __name__=='__main__':
    args = init_argparser()
    main(args)