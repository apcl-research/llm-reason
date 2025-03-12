import json
import argparse
import pickle
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train-filename', type=str, default='/nfs/projects/llm_reason/codegen/concode/train.json')
    parser.add_argument('--val-filename', type=str, default='/nfs/projects/llm_reason/codegen/concode/dev.json')
    parser.add_argument('--test-filename', type=str, default='/nfs/projects/llm_reason/codegen/concode/dev.json')

    args = parser.parse_args()
    train_filename = args.train_filename
    val_filename = args.val_filename
    test_filename = args.test_filename

    newdat_train = list()
    newdat_val = list()
    newdat_test = list()
    traindata = []
    valdata = []
    testdata = []


    with open(train_filename, 'r') as file:
        for line in file:
            traindata.append(json.loads(line))
    with open(val_filename, 'r') as file:
        for line in file:
            valdata.append(json.loads(line))
    with open(test_filename, 'r') as file:
        for line in file:
            testdata.append(json.loads(line))
    
    count_train = 0
    count_val = 0
    count_test = 0
    for data in tqdm(traindata[:]):
        comdat = data["nl"]
        tdat = data["code"]
        samp = {}
        samp['fid'] = count_train
        samp['input'] = f'Please generate a single Java method without import library or class name based on the description {comdat}. Remember just generate a single Java method'
        samp['output'] = f'<s>\t{tdat}</s>'
        newdat_train.append(samp)
        count_train += 1

    for data in tqdm(valdata[:1000]):
        comdat = data["nl"]
        tdat = data["code"]
        samp = {}
        samp['fid'] = count_val
        samp['input'] = f'Please generate a single java method without import library or class name based on the description {comdat}i. Remeber just generate a single Java method.'
        samp['output'] = f'<s>\t{tdat}</s>'
        newdat_val.append(samp)
        count_val += 1
    
    for data in tqdm(testdata[1000:2000]):
        comdat = data["nl"]
        tdat = data["code"]
        samp = {}
        samp['fid'] = count_test
        samp['input'] = f'Please generate a single java method without import library or class name based on the description {comdat}. Remember just generate a single Java method'
        samp['output'] = f'<s>\t'
        newdat_test.append(samp)
        count_test += 1


    
    newdats_train = json.dumps(newdat_train, indent=2)
    newdats_val = json.dumps(newdat_val, indent=2)
    newdats_test = json.dumps(newdat_test, indent=2)
    print(f"training samples: {count_train}")
    print(f"val samples: {count_val}")
    #print(f"test samples: {count_test}")
    with open("data/codegen_train.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_train)
    with open("data/codegen_test.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_test)
    with open("data/codegen_val.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_val)
