import json
import argparse
import pickle
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train-filename', type=str, default='/nfs/projects/llm_reason/code-code/c/train.c-rust.json')
    parser.add_argument('--test-filename', type=str, default='/nfs/projects/llm_reason/code-code/c/test.c-rust.json')
    parser.add_argument('--val-filename', type=str, default='/nfs/projects/llm_reason/code-code/c/valid.c-rust.json')
    #parser.add_argument('--com-filename', type=str, default='/nfs/projects/funcom/data/javastmt_fc/output/coms.val')

    args = parser.parse_args()
    train_filename = args.train_filename
    test_filename = args.test_filename
    val_filename = args.val_filename

    newdat_train = list()
    newdat_val = list()
    newdat_test = list()

    traindata = []
    valdata = []
    testdata = []

    with open(train_filename, 'r') as file:
        alldata = json.load(file)
        for data in alldata:
            traindata.append(data)
    
    with open(val_filename, 'r') as file:
        alldata = json.load(file)
        for data in alldata:
            valdata.append(data)

    with open(test_filename, 'r') as file:
        alldata = json.load(file)
        for data in alldata:
            testdata.append(data)
    
    count_train = 0
    count_val = 0
    count_test = 0
    ref = []
    for data in tqdm(traindata[:]):
        fid = data["index"]
        c_code = data["before"]
        rust_code = data["after"]
        samp = {}
        samp['fid'] = fid 
        samp['input'] = f'[INST] Please translate the given C code {c_code} to Rust[\INST]'
        samp['output'] = f'<s>\t{rust_code}</s>'
        newdat_train.append(samp)
        count_train += 1

    for data in tqdm(valdata[:]):
        fid = data["index"]
        c_code = data["before"]
        rust_code = data["after"]
        samp = {}
        samp['fid'] = fid 
        samp['input'] = f'[INST] Please translate the given C code {c_code} to Rust [\INST]'
        samp['output'] = f'<s>\t{rust_code}</s>'
        newdat_val.append(samp)
        count_val += 1
    
    for data in tqdm(testdata[:]):
        fid = data["index"]
        c_code = data["before"]
        rust_code = data["after"]
        ref.append(rust_code)

        samp = {}
        samp['fid'] = fid 
        samp['input'] = f'Please translate the given C code {c_code} to Rust'
        samp['output'] = f'<s>\t{rust_code}</s>'

        newdat_test.append(samp)
        count_test += 1
with open('ref/codetrans_c_ref.txt', 'w') as file:
    for index, summary in enumerate(ref):
        file.write(f'{index}<SEP>{summary}\n')
    
    #for data in testsets[:]:
    #    count_test += 1
    #    label = data['label']
    #    code = data['code']
    #    samp['fid'] = data['fid']
    #    samp['input'] = f'what are the three most important statements for the privacy label {label} in {code} and only in the code not the random statment?\n<s>'
    #    samp['output'] = f'<s>'
            
    #    newdat_test.append(samp)
   

    #newdat_train.extend(newdat_val)

    newdats_train = json.dumps(newdat_train, indent=2)
    newdats_val = json.dumps(newdat_val, indent=2)
    newdats_test = json.dumps(newdat_test, indent=2)
    print(f"training samples: {count_train}")
    print(f"val samples: {count_val}")
    print(f"test samples: {count_test}")
    with open("data/codetrans_c_train.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_train)
    with open("data/codetrans_c_test.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_test)
    with open("data/codetrans_c_val.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_val)
