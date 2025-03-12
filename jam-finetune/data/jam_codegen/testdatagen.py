# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
from datasets import Dataset

import pickle
import random
import argparse
import bincomb
import os
import json

random.seed(1337)

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-proc', type=int, default=4)
    parser.add_argument('--test-file', type=str, default='/nfs/projects/llm_reason/codegen/concode/dev.json')
    parser.add_argument('--data-dir', type=str, default='codegen_test/')

    args = parser.parse_args()

    num_proc = outdir = args.num_proc
    test_file = args.test_file
    data_dir = args.data_dir
    
    testdata = []
    codelist = []
    with open(test_file, 'r') as file:
        for line in file:
            testdata.append(json.loads(line))

    
    if (not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    count = 0
    for data in tqdm(testdata[1000:2000]):
        com = data["nl"]
        code = data["code"]
        try:
            with open(f'{data_dir}{count}.txt', 'w') as f:
                f.write(f'COM:\t{com}\nTDAT:\t' )
            codelist.append(code)
        except KeyError:
            continue
        count += 1
    count = 0
    with open(f'answer.txt', 'w') as f:
        for index, code in enumerate(codelist):
            f.write(f'{count}<SEP>{code}\n')
            count += 1







