import json
import argparse
import pickle
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train-filename', type=str, default='./data/code-code/train.json')
    parser.add_argument('--val-filename', type=str, default='./data/code-code/dev.json')
    parser.add_argument('--test-filename', type=str, default='./data/code-code/dev.json')

    parser.add_argument('--valjava-file', type=str, default='/nfs/projects/llm_reason/code-code/valid.java-cs.txt.java')
    parser.add_argument('--valcs-file', type=str, default='/nfs/projects/llm_reason/code-code/valid.java-cs.txt.cs')
    parser.add_argument('--trainjava-file', type=str, default='/nfs/projects/llm_reason/code-code/train.java-cs.txt.java')
    parser.add_argument('--traincs-file', type=str, default='/nfs/projects/llm_reason/code-code/train.java-cs.txt.cs')
    parser.add_argument('--test-java-file', type=str, default='/nfs/projects/llm_reason/code-code/test.java-cs.txt.java')
    parser.add_argument('--test-cs-file', type=str, default='/nfs/projects/llm_reason/code-code/test.java-cs.txt.cs')

    args = parser.parse_args()
    valjava_file = args.valjava_file
    valcs_file = args.valcs_file
    trainjava_file = args.trainjava_file
    traincs_file = args.traincs_file
    test_java_file = args.test_java_file
    test_cs_file = args.test_cs_file


    newdat_train = list()
    newdat_val = list()
    newdat_test = list()
    traindata = []
    trainjavadata = []
    traincsdata = []
    valjavadata = []
    valcsdata = []
    testjavadata = []
    testcsdata = []


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
    with open(test_java_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            testjavadata.append(line)
    with open(test_cs_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            testcsdata.append(line)

    count_train = 0
    count_val = 0
    count_test = 0

    for index, javacode in tqdm(enumerate(trainjavadata[:])):
        cscode = traincsdata[index]
        samp = {}
        samp['fid'] = count_train
        samp['input'] = f'[INST] Please translate the given java code {javacode} to c sharp [\INST]'
        samp['output'] = f'<s>\t{cscode}</s>'
        newdat_train.append(samp)
        count_train += 1

    for index, javacode in tqdm(enumerate(valjavadata[:])):
        cscode = valjavadata[index]
        samp = {}
        samp['fid'] = count_val
        samp['input'] = f'[INST] Please translate the given java code {javacode} to c sharp [\INST]'
        samp['output'] = f'<s>\t{cscode}</s>'
        newdat_val.append(samp)
        count_val += 1
    
    for index, javacode in tqdm(enumerate(testjavadata[:])):
        samp = {}
        samp['fid'] = count_test
        samp['input'] = f'Please translate the given java code {javacode} to c sharp'
        samp['output'] = f'<s>\t'
        newdat_test.append(samp)
        count_test += 1


    

    newdats_train = json.dumps(newdat_train, indent=2)
    newdats_val = json.dumps(newdat_val, indent=2)
    newdats_test = json.dumps(newdat_test, indent=2)
    print(f"training samples: {count_train}")
    print(f"val samples: {count_val}")
    #print(f"test samples: {count_test}")
    with open("data/codetrans_train.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_train)
    with open("data/codetrans_test.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_test)
    with open("data/codetrans_val.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_val)
