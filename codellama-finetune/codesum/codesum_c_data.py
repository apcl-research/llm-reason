import json
import argparse
import pickle
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--filename', type=str, default='/nfs/projects/llm_reason/codesum/c/c_functions_all_data.jsonl')
    #parser.add_argument('--com-filename', type=str, default='/nfs/projects/funcom/data/javastmt_fc/output/coms.val')

    args = parser.parse_args()
    filename = args.filename

    newdat_train = list()
    newdat_val = list()
    newdat_test = list()

    alldata = []
    with open(filename, 'r') as file:
        for line in file:
            alldata.append(json.loads(line))
    train_size = int(len(alldata)*0.6)
    val_test_size = int(len(alldata)*0.2)

    traindata = alldata[:train_size]
    valdata = alldata[len(traindata):len(traindata) + val_test_size]
    testdata = alldata[len(traindata)+ len(valdata):]

    
    count_train = 0
    count_val = 0
    count_test = 0
    ref = []
    for data in tqdm(traindata[:]):
        summary = data["summary"]
        function = data["function"]
        samp = {}
        samp['fid'] = count_train
        samp['input'] = f'[INST] Please use one senternce to describe the given C method {function}.[\INST]'
        samp['output'] = f'<s>\t{summary}</s>'
        newdat_train.append(samp)
        count_train += 1

    for data in tqdm(valdata[:]):
        summary = data["summary"]
        function = data["function"]
        samp = {}
        samp['fid'] = count_val
        samp['input'] = f'[INST] Please use one senternce to describe the given C method {function}.[\INST]'
        samp['output'] = f'<s>\t{summary}</s>'
        newdat_val.append(samp)
        count_val += 1
    
    for data in tqdm(testdata[:]):
        summary = data["summary"]
        ref.append(summary)
        function = data["function"]
        samp = {}
        samp['fid'] = count_test
        samp['input'] = f'Please use one senternce to describe the given C method {function}.'
        samp['output'] = f'<s>\t'
        newdat_test.append(samp)
        count_test += 1
with open('ref/codesum_c_ref.txt', 'w') as file:
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
    with open("data/codesum_c_train.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_train)
    with open("data/codesum_c_test.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_test)
    with open("data/codesum_c_val.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_val)
