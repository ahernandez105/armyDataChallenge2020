import os
import argparse
import pdb

FY_2019_Q_COL = 10
SKU_COL = 0

def init_argparser():
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, required=False, default='data/csv/Q Score Summary.txt')
    args.add_argument('--skus', type=str, required=False, default='data/csv/train_skus.txt')
    args.add_argument('--dest', type=str, required=False, default='data/train/skus')

    return args.parse_args()

def main(args):
    # read the original skus
    in_skus = []
    with open(args.skus, 'r') as skus:
        for _, line in enumerate(skus):
            line = line.strip()
            if len(line)>0:
                in_skus.append(line)
    in_skus = set(in_skus)

    # identify skus that are greater than 0 w.r.t to fy 2019 q value
    skus = []
    with open(args.data, 'r') as file:
        file.readline()
        for row, line in enumerate(file):
            line = line.replace('"',"").split(',')
            if line[SKU_COL] in in_skus:
                if len(line[FY_2019_Q_COL])==0:
                    continue
                if float(line[FY_2019_Q_COL])<0:
                    continue
                skus.append(line[SKU_COL])
    
    # write the out file
    with open(args.dest, 'w') as dest:
        dest.write("\n".join(skus))

if __name__=='__main__':
    args = init_argparser()
    main(args)
