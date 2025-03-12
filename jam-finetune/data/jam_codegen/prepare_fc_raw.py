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
import json

random.seed(1337)

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-proc', type=int, default=4)
    parser.add_argument('--val-file', type=str, default='/nfs/projects/llm_reason/codegen/concode/dev.json')
    parser.add_argument('--train-file', type=str, default='/nfs/projects/llm_reason/codegen/concode/train.json')
    parser.add_argument('--data-dir', type=str, default='bins/')
    args = parser.parse_args()

    num_proc = outdir = args.num_proc
    val_file = args.val_file
    train_file = args.train_file
    data_dir = args.data_dir
    traindata = [] 
    valdata = [] 
    with open(train_file, 'r') as file:
        for line in file:
            traindata.append(json.loads(line))
    with open(val_file, 'r') as file:
        for line in file:
            valdata.append(json.loads(line))
        #data = json.load(file)
    count = 0 
    count_val = 0
    for partnum in range(0, 1):

        print(f'starting part {partnum}')

        txtfiles = list()
        txtfiles_val = list()
        bin_file_path = data_dir + f'/val_2pt_p{partnum}.bin'

        #if os.path.isfile(bin_file_path):
        #    continue



        for data in tqdm(traindata[:]):

            with open(f'tmp/{count}', 'w') as f:
                comdat = data["nl"]
                tdat = data["code"]
                f.write(f'COM:\t{comdat}\nTDAT:\t{tdat}' )
                count += 1
            txtfiles.append(f'tmp/{count}')
        for data in tqdm(valdata[:1000]):

            with open(f'tmp/{count}', 'w') as f:
                comdat = data["nl"]
                tdat = data["code"]
                f.write(f'COM:\t{comdat}\nTDAT:\t{tdat}' )
                count_val += 1
            txtfiles_val.append(f'tmp/{count}')
        
        dataset = load_dataset('text', data_files={'train': txtfiles, 'val':txtfiles_val}, sample_by="document")
        
        shmdir = 'tmp/'
        for f in os.listdir(shmdir):
            os.remove(os.path.join(shmdir, f))

        pickle.dump(dataset, open(f'pkls/dataset_funcom_2pt_p{partnum}.pkl', 'wb'))



        # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
        enc = tiktoken.get_encoding("gpt2")
        def process(example):
            ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
            ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
            # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
            out = {'ids': ids, 'len': len(ids)}
            return out

        # tokenize the dataset
        tokenized = dataset.map(
            process,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'])
            filename = os.path.join(data_dir, f'{split}_2pt_p{partnum}.bin')
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

            print(f"writing {filename}...")
            idx = 0
            for example in tqdm(dset):
                arr[idx : idx + example['len']] = example['ids']
                idx += example['len']
            arr.flush()
    
    bincomb.main('bins/')
