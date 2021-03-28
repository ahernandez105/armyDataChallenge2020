import os
import sys
import argparse
import pdb
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from util import preprocess

COLS = ["ASSET_UID", "DATAYEAR", "MISSION_SC"]

def init_argparser():
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, required=False, default='data/csv/ISR MODEL Data 2014 to 2019.txt')
    args.add_argument('--skus', type=str, required=False, default='data/train/skus')
    args.add_argument('--no_nan', action='store_true', required=False)
    args.add_argument('--dest', type=str, required=False, default='data/train/mission_sc_all')

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
        pd.pivot_table(df, values='MISSION_SC', index='Asset_UID', columns='DATAYEAR', 
        aggfunc=sum, fill_value=-1).to_records())

    return df

def main(args):
    df = preprocess.sort_df_on_skus(read_data(args.data), preprocess.get_skus_df(args.skus))
    print(df.shape)
    print('MEANS')
    print(df.mean(skipna=True))
    print('STDs')
    print(df.std(skipna=True))
    df = df.fillna(-1)
    df = df.applymap(lambda x: x if x>=0 else -1)
    values = df[['20144.0', '20154.0', '20164.0', '20174.0', '20184.0', '20194.0']].values

    lines = []
    if args.no_nan:
        for vec in values:
            mask = np.zeros(6, bool)
            mask[np.where(vec >= 0)[0]] = True
            vec = vec[mask]

            if(len(vec)==0):
                lines.append('-1')
            else:
                lines.append(" ".join([str(x) for x in vec]))
    else:
        for vec in values:
            lines.append(" ".join([str(x) for x in vec]))
    
    with open(args.dest, 'w') as file:
        file.write("\n".join(lines))

if __name__=='__main__':
    args = init_argparser()
    main(args)