import json
import argparse
import pickle
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--fundats-filename', type=str, default='/nublar/datasets/jm52m/fundats-j1.pkl')
    parser.add_argument('--testfids-filename', type=str, default='/nublar/datasets/jm52m/raw_data/jam-cgpt-testfid.pkl')
    parser.add_argument('--valfids-filename', type=str, default='/nublar/datasets/jm52m/raw_data/jam-cgpt-valfid.pkl')
    parser.add_argument('--coms-filename', type=str, default='/nublar/datasets/jm52m/raw_data/jam-cgpt-raw170k.pkl')

    args = parser.parse_args()
    fundats_filename = args.fundats_filename
    testfids_filename = args.testfids_filename
    valfids_filename = args.valfids_filename
    coms_filename = args.coms_filename

    newdat_train = list()
    newdat_val = list()
    newdat_test = list()
    
    fundats = pickle.load(open(fundats_filename, "rb"))
    testfids = pickle.load(open(testfids_filename, "rb"))
    valfids = pickle.load(open(valfids_filename, "rb"))
    allcoms = pickle.load(open(coms_filename, "rb"))
    count_train = 0
    count_val = 0
    count_test = 0
    for fid in tqdm(list(allcoms.keys())):
        code = fundats[fid]
        coms = allcoms[fid]
        samp = dict()
        if(fid in testfids):
            count_test += 1
            samp['fid'] = fid
            samp['input'] = f'Please use one sentence to describe the given method {code}'
            samp['output'] = f'<s>\t'
            newdat_test.append(samp)

        elif(fid in valfids):
            count_val += 1
            samp['input'] = f'[INST] Please use one sentence to describe the given method {code} [\INST]'
            samp['output'] = f'<s>\t{coms}<\s>'
            newdat_val.append(samp)
        else:
            count_train += 1

            samp['input'] = f'[INST] Please use one sentence to describe the given method {code} [\INST]'
            samp['output'] = f'<s>\t{coms}<\s>'
            
            newdat_train.append(samp)

    newdats_train = json.dumps(newdat_train, indent=2)
    newdats_val = json.dumps(newdat_val, indent=2)
    newdats_test = json.dumps(newdat_test, indent=2)
    print(f"training samples: {count_train}")
    print(f"val samples: {count_val}")
    #print(f"test samples: {count_test}")
    with open("data/cgpt_170k_train.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_train)
    with open("data/cgpt_170k_test.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_test)
    with open("data/cgpt_170k_val.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_val)

