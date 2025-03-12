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

random.seed(1337)

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-proc', type=int, default=4)
    parser.add_argument('--data-file', type=str, default='/nfs/dropbox/llm_reason/data_analysis/data_flow_results_java_test.pkl')
    parser.add_argument('--data-dir', type=str, default='dataflow/')

    args = parser.parse_args()

    num_proc = args.num_proc
    data_file = args.data_file
    data_dir = args.data_dir

    alldata = pickle.load(open(data_file, 'rb'))

    if (not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    for data in tqdm(alldata[:]):
        main_method_name = data["main_method_name"]
        source_code = data["code"]
        tools_results = data["results"]
        sinks = []
        
        for result in tools_results:
            sinks.extend(list(result.keys()))
        for sink in sinks:
            try:
                with open(f'{data_dir}{main_method_name}_{sink}.txt', 'w') as f:
                    f.write(f'TDAT:\t{source_code}\nSOUCE:\t{main_method_name}\nSINK:\t{sink}\nRESULT:\t')
            except KeyError:
                continue
