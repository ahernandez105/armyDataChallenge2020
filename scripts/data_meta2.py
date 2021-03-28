import os
import sys
import argparse
import pdb
import re
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from util import preprocess

COLS = ['DATAYEAR', 'ASSET_UID', 'FACNO', 'RPA_OPERAT', 'RPA_TYPE_C','RPA_INTERE', 'CONSTRUCTI']
PATTERN = '[0-9]'

def init_argparser():
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, required=False, default='data/csv/ISR MODEL Data 2014 to 2019.txt')
    args.add_argument('--skus', type=str, required=False, default='data/train/skus')
    args.add_argument('--dest', type=str, required=False, default='data/train/meta2')

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
        pd.pivot_table(df, values=[x for x in COLS if x not in ['ASSET_UID', 'DATAYEAR']],
        index='Asset_UID', columns='DATAYEAR', aggfunc=lambda x: x).to_records())
    
    return df

def func(x):
    for i in reversed(range(6)):
        if x[i] is not None:
            return str(x[i]).replace(" ", "-")
    
    return 'None'

def func_facno(x):
    for i in reversed(range(6)):
        if x[i] is not None:
            string = str(x[i]).replace(" ",'-')
            temp_string = re.sub(PATTERN, '', string)
            if len(temp_string)==0:
                return str(round(int(string),-2))
            else:
                return temp_string
    
    return 'None'

def main(args):
    df = preprocess.sort_df_on_skus(read_data(args.data), preprocess.get_skus_df(args.skus))

    cols = ["('{c}', {y})".format(c='FACNO', y=x) for x in ['20144.0', '20154.0', '20164.0', '20174.0', '20184.0', '20194.0']]
    df['FACNO'] = df[cols].apply(func_facno, axis=1)

    for col in COLS[3:]:
        cols = ["('{c}', {y})".format(c=col, y=x) for x in ['20144.0', '20154.0', '20164.0', '20174.0', '20184.0', '20194.0']]
        df[col] = df[cols].apply(func, axis=1)
    
    lines = []
    values = df[COLS[2:]].values
    for vec in values:
        lines.append(" ".join(vec))
    
    with open(args.dest, 'w') as file:
        file.write("\n".join(lines))


if __name__=='__main__':
    main(init_argparser())