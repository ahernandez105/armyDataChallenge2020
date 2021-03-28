import os
import sys
import argparse
import pdb
import re
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from util import preprocess

COLS = ['DATAYEAR', 'ASSET_UID', 'QIC']
PATTERN = '[0-9]'
COLS_ = ['20144.0', '20154.0', '20164.0', '20174.0', '20184.0', '20194.0']

def init_argparser():
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, required=False, default='data/csv/ISR MODEL Data 2014 to 2019.txt')
    args.add_argument('--skus', type=str, required=False, default='data/train/skus')
    args.add_argument('--dest', type=str, required=False, default='data/train/qic')

    return args.parse_args()

def asset_id(x):
    try:
        return int(x)
    except:
        return -1

def read_data(path):
    df = pd.read_csv(path, header=0, usecols=COLS)
    df.rename(columns={"ASSET_UID":'Asset_UID'}, inplace=True)
    df[['Asset_UID']] = df[['Asset_UID']].applymap(lambda x: asset_id(x))
    df = pd.DataFrame(
        pd.pivot_table(df, values='QIC',
        index='Asset_UID', columns='DATAYEAR', aggfunc=lambda x: x).to_records())
    
    return df

def func(x):
    try:
        x = float(x)
        if x < 0:
            return np.nan
        else:
            return x
    except:
        return np.nan

def main(args):
    df = preprocess.sort_df_on_skus(read_data(args.data), preprocess.get_skus_df(args.skus))
    for c in COLS_:
        df[c] = df[[c]].apply(func, axis=1)
        print('MEAN '+c)
        print(df[c].mean(axis=0, skipna=True))
        print('STD'+c)
        print(df[c].std(axis=0, skipna=True))

    df = df.fillna(-1)
    lines = []
    values = df[COLS_].values
    for vec in values:
        lines.append(' '.join([str(x) for x in vec]))
    
    with open(args.dest,'w') as file:
        file.write("\n".join(lines))

if __name__=='__main__':
    main(init_argparser())