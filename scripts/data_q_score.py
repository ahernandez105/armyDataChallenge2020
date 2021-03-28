import os
import sys
import argparse
import pdb
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from util import preprocess

COLS_Q = [
    'Asset_UID', 'FY14 Asset Level Data_Qual_Score', 'FY15 Asset Level Data_Qual_Score', 
    'FY16 Asset Level Data_Qual_Score', 'FY17 Asset Level Data_Qual_Score', 
    'FY18 Asset Level Data_Qual_Score', 'FY19 Asset Level Data_Qual_Score']

COLS_R = [
    'FY14 Asset Level Data_Rating_Method', 'FY15 Asset Level Data_Rating_Method',
    'FY16 Asset Level Data_Rating_Method', 'FY17 Asset Level Data_Rating_Method',
    'FY18 Asset Level Data_Rating_Method', 'FY19 Asset Level Data_Rating_Method', 'Rating_Method']

def init_argparser():
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, required=False, default='data/csv/Q Score Summary.txt')
    args.add_argument('--skus', type=str, required=False, default='data/train/skus')
    args.add_argument('--no_nan', action='store_true', required=False)
    args.add_argument('--dest_q_scores', type=str, required=False, default='data/train/q_score_all')
    args.add_argument('--dest_ratings', type=str, required=False, default='data/train/ratings_all')

    return args.parse_args()

def print_metrics(df, args, kind):
    df = preprocess.sort_df_on_skus(df, preprocess.get_skus_df(args.skus))
    if kind=='mean':
        print('MEANS')
        print(df[COLS_Q[1:]].mean(axis=0, skipna=True))
    else:
        print('STDS')
        print(df[COLS_Q[1:]].std(axis=0, skipna=True))


def read_data(args):
    df = pd.read_csv(args.data, header=0, usecols=COLS_Q + COLS_R)
    df[COLS_Q[1:]] = df[COLS_Q[1:]].applymap(lambda x: x if (x>=0 or pd.isnull(x)) else -1)
    df[COLS_Q[1:]]=df[COLS_Q[1:]].fillna(-1)
    df[COLS_R]=df[COLS_R].fillna("<UNK>")
    df[COLS_R] = df[COLS_R].applymap(lambda x: "Business-Rule" if x=="Business Rule" else x)

    return df

def main(args):
    df = preprocess.sort_df_on_skus(read_data(args), preprocess.get_skus_df(args.skus))
    q_values = df[COLS_Q[1:]].values
    r_values = df[COLS_R].values

    q_lines, r_lines = [], []
    if args.no_nan:
        for i in range(len(q_values)):
            mask = np.zeros(len(COLS_R), bool)
            mask[np.where(q_values[i] >= 0)[0]] = True
            ex_q_values = q_values[i][mask]
            ex_r_values = r_values[i][mask]

            if(ex_q_values.shape[0]==0):
                q_lines.append(str(-1))
                r_lines.append("<UNK>")
            else:
                q_lines.append(" ".join([str(x) for x in ex_q_values]))
                r_lines.append(" ".join(ex_r_values))
                
    else:
        for i in range(len(q_values)):
            q_lines.append(" ".join([str(x) for x in q_values[i]]))
            r_lines.append(" ".join(r_values[i]))
            
    with open(args.dest_q_scores, 'w') as file:
        file.write("\n".join(q_lines))
    
    with open(args.dest_ratings, 'w') as file:
        file.write("\n".join(r_lines))

if __name__=='__main__':
    args = init_argparser()
    main(args)


