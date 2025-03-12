import json
import argparse
import pickle
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train-filename', type=str, default='/nfs/projects/llm_reason/codegen/c_c++_train.pkl')
    parser.add_argument('--test-filename', type=str, default='/nfs/projects/llm_reason/codegen/c_c++_test.pkl')
    parser.add_argument('--val-filename', type=str, default='/nfs/projects/llm_reason/codegen/c_c++_val.pkl')

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

    traindata = pickle.load(open(train_filename, "rb"))
    valdata = pickle.load(open(val_filename, "rb"))
    testdata = pickle.load(open(test_filename, "rb"))
    
    count_train = 0
    count_val = 0
    count_test = 0
    ref = []
    for data in tqdm(traindata[:]):
        text = data["text"]
        source_code = data["code"]
        samp = {}
        samp['fid'] = count_train 
        samp['input'] = f'[INST] Please generate a single C method without import library or class name based on the description {text}. Remember just generate a single C method[\INST]'
        samp['output'] = f'<s>\t{source_code}</s>'
        newdat_train.append(samp)
        count_train += 1

    for data in tqdm(valdata[:]):
        text = data["text"]
        source_code = data["code"]
        samp = {}
        samp['fid'] = count_val 
        samp['input'] = f'[INST] Please generate a single C method without import library or class name based on the description {text}. Remember just generate a single C method [\INST]'
        samp['output'] = f'<s>\t{source_code}</s>'
        newdat_val.append(samp)
        count_val += 1
    
    for data in tqdm(testdata[:]):
        text = data["text"]
        source_code = data["code"]
        ref.append(source_code)

        samp = {}
        samp['fid'] = count_test 
        samp['input'] = f'[INST] Please generate a single C method without import library or class name based on the description {text}. Remember just generate a single C method [\INST]'
        samp['output'] = f'<s>\t{source_code}</s>'

        newdat_test.append(samp)
        count_test += 1
with open('ref/codegen_c_ref.txt', 'w') as file:
    for index, summary in enumerate(ref):
        file.write(f'{index}<SEP>{summary}\n')
    

    newdats_train = json.dumps(newdat_train, indent=2)
    newdats_val = json.dumps(newdat_val, indent=2)
    newdats_test = json.dumps(newdat_test, indent=2)
    print(f"training samples: {count_train}")
    print(f"val samples: {count_val}")
    print(f"test samples: {count_test}")
    with open("data/codegen_c_train.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_train)
    with open("data/codegen_c_test.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_test)
    with open("data/codegen_c_val.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_val)
