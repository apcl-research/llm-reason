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
    parser.add_argument('--valjava-file', type=str, default='/nfs/projects/llm_reason/code-code/valid.java-cs.txt.java')
    parser.add_argument('--valcs-file', type=str, default='/nfs/projects/llm_reason/code-code/valid.java-cs.txt.cs')
    parser.add_argument('--trainjava-file', type=str, default='/nfs/projects/llm_reason/code-code/train.java-cs.txt.java')
    parser.add_argument('--traincs-file', type=str, default='/nfs/projects/llm_reason/code-code/train.java-cs.txt.cs')
    parser.add_argument('--data-dir', type=str, default='bins/')
    args = parser.parse_args()

    num_proc = outdir = args.num_proc
    valjava_file = args.valjava_file
    valcs_file = args.valcs_file
    trainjava_file = args.trainjava_file
    traincs_file = args.traincs_file
    data_dir = args.data_dir
    trainjavadata = [] 
    traincsdata = [] 
    valjavadata = [] 
    valcsdata = [] 

    with open(trainjava_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            trainjavadata.append(line)
    with open(traincs_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            traincsdata.append(line)
    with open(valjava_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            valjavadata.append(line)
    with open(valcs_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            valcsdata.append(line)
    


    count = 0 
    count_val = 0
    for partnum in range(0, 1):

        print(f'starting part {partnum}')

        txtfiles = list()
        txtfiles_val = list()
        bin_file_path = data_dir + f'/val_2pt_p{partnum}.bin'

        #if os.path.isfile(bin_file_path):
        #    continue



        for index, javacode in tqdm(enumerate(trainjavadata[:])):
            cscode = traincsdata[index]
            #print(javacode)
            #print("----", cscode)
            with open(f'tmp/{count}', 'w') as f:
                f.write(f'JAVA:\t{javacode}\nCSHARP:\t{cscode}' )
                count += 1
            txtfiles.append(f'tmp/{count}')

        for index, javacode in tqdm(enumerate(valjavadata[:])):
            cscode = valjavadata[index]
            with open(f'tmp/{count}', 'w') as f:
                f.write(f'JAVA:\t{javacode}\nCSHARP:\t{cscode}' )
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
