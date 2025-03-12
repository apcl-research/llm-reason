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

random.seed(1337)

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-proc', type=int, default=4)
    parser.add_argument('--data-file', type=str, default='/nfs/projects/llm_reason/data_analysis/data_flow_results_java_new.pkl')
    parser.add_argument('--data-dir', type=str, default='bins/')
    args = parser.parse_args()

    num_proc = outdir = args.num_proc
    data_file = args.data_file
    data_dir = args.data_dir

    data_analysis_results = pickle.load(open(data_file, 'rb'))

    #fundats_fids = list(fundats.keys())
    filtered_results = []
    main_method_name_list = []
    for data in data_analysis_results:
        if(data["main_method_name"] in main_method_name_list):
            continue
        else:
            main_method_name_list.append(data["main_method_name"])
        results = data["results"]
        if(results == []):
            continue
        result = results[0]
        method_name = list(result.keys())[0]
        if(method_name.strip() =="" and len(results) == 1):
            continue
        count_empty = 0
        for res in results:
            method_name = list(res.keys())[0]
            if(res[method_name] == []):
                count_empty += 1
        if(count_empty == len(results)):
            continue
        filtered_results.append(data)

    print("dddddddddddddddddddd", len(filtered_results))
    train_set = filtered_results[:15000]
    val_set = filtered_results[15000:]
    pt = int(len(filtered_results) * 1)
    count = 0 
    count_val = 0
    for partnum in range(0, 1):

        print(f'starting part {partnum}')

        txtfiles = list()
        txtfiles_val = list()
        bin_file_path = data_dir + f'/val_2pt_p{partnum}.bin'

        if os.path.isfile(bin_file_path):
            continue

        start_pt = (partnum * pt)
        end_pt = ((partnum+1) * pt)

        count_data = 0

        for data in tqdm(filtered_results[:]):
            code = data["code"]
            main_method_name = data["main_method_name"]
            allresults = data["results"]
            for results in allresults:
                
                method_name = list(results.keys())[0]
                if(method_name ==""):
                    continue
                if(results[method_name] == []):
                    continue
                dataflow = ""
                temp_results = [] 
                for result in results[method_name]:
                    if('->'.join(result) not in temp_results):
                        temp_results.append('->'.join(result))
                        dataflow += '->'.join(result)
                        dataflow += '\n'
                if(count_data <= 15000):
                    with open(f'tmp/{count_data}', 'w') as f:
                        f.write(f'TDAT:\t{code}\nSOURCE:\t{main_method_name}\nSINK:\t{method_name}\nRESULT:\t{dataflow}')
                        txtfiles.append(f'tmp/{count_data}')
                else:
                    with open(f'tmp/{count_data}', 'w') as f:
                        f.write(f'TDAT:\t{code}\nSOURCE:\t{main_method_name}\nSINK:\t{method_name}\nRESULT:\t{dataflow}')
                        txtfiles_val.append(f'tmp/{count_data}')

            count_data += 1
        
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
