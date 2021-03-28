import os
import sys
import argparse
import pdb
import shutil
from pathlib import Path
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from util import preprocess

def init_argparser():
    args = argparse.ArgumentParser()
    args.add_argument('--folder', type=str, default='data/train')

    return args.parse_args()

def main(args):
    dest_folder = os.path.join(args.folder, 'datasets')
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.mkdir(dest_folder)

    for path in os.listdir(args.folder):
        if Path(path).name != 'README' and path != 'datasets':
            os.mkdir(os.path.join(dest_folder, Path(path).name))
            with open(os.path.join(args.folder, path), 'r') as file:
                for i, line in enumerate(file):
                    out = open(os.path.join(dest_folder, Path(path).name, str(i)), 'w')
                    out.write(line)
                    out.close()

if __name__=='__main__':
    main(init_argparser())

