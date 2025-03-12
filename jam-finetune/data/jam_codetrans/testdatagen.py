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
    parser.add_argument('--test-java-file', type=str, default='/nfs/projects/llm_reason/code-code/test.java-cs.txt.java')
    parser.add_argument('--test-cs-file', type=str, default='/nfs/projects/llm_reason/code-code/test.java-cs.txt.cs')
    parser.add_argument('--data-dir', type=str, default='codetrans_test/')

    args = parser.parse_args()

    num_proc = outdir = args.num_proc
    test_java_file = args.test_java_file
    test_cs_file = args.test_cs_file
    data_dir = args.data_dir
    
    testjavadata = []
    testcsdata = []

    with open(test_java_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            testjavadata.append(line)
    with open(test_cs_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            testcsdata.append(line)



    
    if (not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    count = 0
    for index, javacode in tqdm(enumerate(testjavadata[:])):

        try:
            with open(f'{data_dir}{count}.txt', 'w') as f:
                f.write(f'JAVA:\t{javacode}\nCSHARP:\t' )
        except KeyError:
            continue
        count += 1
    count = 0
    with open(f'codetrans.txt', 'w') as f:
        for index, code in enumerate(testcsdata):
            f.write(f'{count}<SEP>{code}\n')
            count += 1







