import json
import argparse
import pickle
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data-filename', type=str, default='/nfs/projects/llm_reason/callgraph/java_new/callgraph_java_train_q90.pkl')
    parser.add_argument('--test-filename', type=str, default='/nfs/projects/llm_reason/callgraph/java_new/callgraph_java_test_q90.pkl')

    args = parser.parse_args()
    data_filename = args.data_filename
    test_filename = args.test_filename

    newdat_train = list()
    newdat_val = list()
    newdat_test = list()
    callgraph_results = pickle.load(open(data_filename, "rb"))
    testset_results =  pickle.load(open(test_filename, "rb"))
    allresults = {}
    for result in callgraph_results:
        fid = result["root"]
        if(fid not in allresults):
            allresults[fid] = []
        allresults[fid].append(result)
    train_fids = list(allresults.keys())[:20000]
    val_fids = list(allresults.keys())[20001:]

    count_train = 0
    count_val = 0
    count_test = 0
    count_one = 0
    
    
    
    for fid in tqdm(train_fids[:]):
        callgraphdata = allresults[fid]
        for dat in callgraphdata:
            fid = dat["root"]
            code = dat["code"]
            method_name = dat["method_name"]
            results = dat["path"]
            callgraph_path = []
            for result in results:
                result = [res.split(".")[-1] for res in result]
                callgraph_path.append('->'.join(result))
            callgraph_path = '\n'.join(callgraph_path)

            samp = dict()
            samp['fid'] = count_train
            samp['input'] = f'[INST] Given the Java code {code}, Please generate the complete callgraph path of the method {method_name} [\INST]'
            samp['output'] = f'<s>\t{callgraph_path}<\s>'
            
            newdat_train.append(samp)

            count_train += 1
        
    for fid in tqdm(val_fids[:]):
        callgraphdata = allresults[fid]
        for dat in callgraphdata:
            fid = dat["root"]
            code = dat["code"]
            method_name = dat["method_name"]
            results = dat["path"]
            callgraph_path = []
            for result in results:
                result = [res.split(".")[-1] for res in result]
                callgraph_path.append('->'.join(result))
            callgraph_path = '\n'.join(callgraph_path)

            samp = dict()
            samp['fid'] = count_val
            samp['input'] = f'[INST] Given the Java code {code}, Please generate the complete callgraph path of the method {method_name} [\INST]'
            samp['output'] = f'<s>\t{callgraph_path}<\s>'

            newdat_val.append(samp)

            count_val += 1
            
    for dat in tqdm(testset_results[:]):
        fid = dat["root"]
        code = dat["code"]
        method_name = dat["method_name"]
        results = dat["path"]
        callgraph_path = []
        for result in results:
            result = [res.split(".")[-1] for res in result]
            callgraph_path.append('->'.join(result))
        callgraph_path = '\n'.join(callgraph_path)

        samp = dict()
        samp["fid"] = count_test 
        samp['input'] = f'[INST] Given the Java code {code}, Please generate the complete callgraph path of the method {method_name} [\INST]'
        samp['output'] = f'<s>\t{callgraph_path}<\s>'

        newdat_test.append(samp)

        
        count_test += 1

    

    newdats_train = json.dumps(newdat_train, indent=2)
    newdats_val = json.dumps(newdat_val, indent=2)
    newdats_test = json.dumps(newdat_test, indent=2)
    print(f"training samples: {count_train}")
    print(f"val samples: {count_val}")
    print(f"test samples: {count_test}")
    print(f"count no data flow : {count_one}")
    with open("data/callgraph_java_train.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_train)
    with open("data/callgraph_java_test.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_test)
    with open("data/callgraph_java_val.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_val)

