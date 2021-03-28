import os
import sys
import argparse
import pdb
import re
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from util import preprocess

COLS = [
    'accountable_sub_org_name', 'facility_book_value','facility_replacement_value',
    'instl_name', 'op_status', 'st_terr_name', 'type_desc', 
    'linear_feet_total', 'acres_total', 'square_feet_total']
STR_COLS = [
    'accountable_sub_org_name', 'instl_name', 'op_status', 'st_terr_name', 'type_desc']
FLOAT_COLS = [
    'facility_book_value','facility_replacement_value', 
    'linear_feet_total', 'acres_total', 'square_feet_total']
ID = ['rpa_uid']

def init_argparser():
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, required=False, default='data/csv/hqiis_data.txt')
    args.add_argument('--skus', type=str, required=False, default='data/train/skus')
    args.add_argument('--dest', type=str, required=False, default='data/train/meta3')

    return args.parse_args()

def asset_id(x):
    try:
        return int(x)
    except:
        return -1

def str_func(x):
    if x==np.nan:
        return x
    else:
        return str(x).replace(" ", "-")

def float_func(x):
    try:
        x = float(x)
        if x < 0:
            return 0
        else:
            return x
    except:
        return np.nan

def read_data(path):
    df = pd.read_csv(path, header=0, usecols=COLS + STR_COLS + ID, sep='\t', engine='python')
    df.rename(columns={"rpa_uid":'Asset_UID'}, inplace=True)
    df[['Asset_UID']] = df[['Asset_UID']].applymap(lambda x: asset_id(x))

    return df

def main(args):
    df = preprocess.sort_df_on_skus(read_data(args.data), preprocess.get_skus_df(args.skus))

    for col in STR_COLS:
        df[col]=df[[col]].applymap(lambda x: str_func(x))    

    for col in FLOAT_COLS:
        df[col]=df[[col]].applymap(lambda x: float_func(x))
        print(col)
        print(df[col].mean(axis=0, skipna=True))
        print(df[col].std(axis=0, skipna=True))
    
    lines = []
    values = df[STR_COLS+FLOAT_COLS].values
    for vec in values:
        lines.append(" ".join([str(x) for x in vec]))
    
    with open(args.dest, 'w') as file:
        file.write("\n".join(lines))

if __name__=='__main__':
    main(init_argparser())