import os
import sys
import argparse
import pdb
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from util import preprocess

COLS = ['Asset_UID', 'Facility_Built_Date', 'Army Head Quarter""', 'Region', 'Base', 'Rating_Method']

def init_argparser():
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, required=False, default='data/csv/Q Score Summary.txt')
    args.add_argument('--skus', type=str, required=False, default='data/train/skus')
    args.add_argument('--dest', type=str, required=False, default='data/train/meta')

    return args.parse_args()

def do_date(x):
    x = str(x).split('/')

    if len(x)==3:
        return int(x[2])
    else:
        return -1

def print_metrics(df, args):
    df = preprocess.sort_df_on_skus(df, preprocess.get_skus_df(args.skus))
    print('MEAN')
    print(df.mean(axis=0))
    print('STD')
    print(df.std(axis=0))


def read_data(args):
    df = pd.read_csv(args.data, header=0, usecols=COLS)
    df[['Facility_Built_Date']] = df[['Facility_Built_Date']].applymap(lambda x: do_date(x))
    df[COLS[2:]] = df[COLS[2:]].applymap(lambda x: str(x).replace(" ", "-"))
    print_metrics(df[COLS[0:2]], args)

    return df

def main(args):
    values = preprocess.sort_df_on_skus(read_data(args), preprocess.get_skus_df(args.skus))[COLS[1:]].values

    lines = []
    for vec in values:
        lines.append(" ".join([str(x) for x in vec]))
    
    with open(args.dest, 'w') as file:
        file.write("\n".join(lines))

if __name__=='__main__':
    args = init_argparser()
    main(args)

