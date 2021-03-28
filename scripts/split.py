import numpy as np
import os
from pathlib import Path
import argparse

def init_argparser():
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, required=False, default='data/csv/Prediction Template.txt')
    args.add_argument('--seed', type=int, required=False, default=1111)

    return args.parse_args()

def write_split(dest, skus, arr):
    with open(dest, 'w') as file:
        for i in arr:
            file.write(skus[i]+'\n')

def main(args):
    skus = []
    with open(args.data, "r") as file:
        file.readline() # skip col headers
        for line in file:
            skus.append(line.split(",")[0].replace('"',""))
    
    arr = np.arange(len(skus))
    np.random.seed(args.seed)
    np.random.shuffle(arr)
    train_end = int(0.90*len(arr))
    valid_end = int(0.95*len(arr))
    dest_dir = str(Path(args.data).parent)

    write_split(dest_dir+'/train_skus.txt', skus, arr[0:train_end])
    write_split(dest_dir+'/valid_skus.txt', skus, arr[train_end:valid_end])
    write_split(dest_dir+'/test_skus.txt', skus, arr[valid_end:])

if __name__=="__main__":
    args = init_argparser()
    main(args)



