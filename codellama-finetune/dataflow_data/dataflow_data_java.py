import json
import argparse
import pickle
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data-filename', type=str, default='/nfs/projects/llm_reason/data_analysis/data_flow_results_java_new.pkl')
    parser.add_argument('--test-filename', type=str, default='/nfs/dropbox/llm_reason/data_analysis/data_flow_results_java_test.pkl')

    args = parser.parse_args()
    data_filename = args.data_filename
    test_filename = args.test_filename

    newdat_train = list()
    newdat_val = list()
    newdat_test = list()

    data_analysis_results = pickle.load(open(data_filename, "rb"))
    
    test_dat = pickle.load(open(test_filename, "rb"))
    

    count_train = 0
    count_val = 0
    count_test = 0
    count_one = 0
    
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

    
    train_dat = filtered_results[:15000]
    val_dat = filtered_results[15000:]
    for dat in tqdm(train_dat[:]):
        code = dat["code"]
        main_method_name = dat["main_method_name"]
        allresults = dat["results"]

        samp = dict()
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

            samp['input'] = f'[INST] Please generate the data flow of the Jave code {code} from source {main_method_name} to sink {method_name} [\INST]'
            samp['output'] = f'<s>\t{dataflow}<\s>'
            
            newdat_train.append(samp)

        count_train += 1

    for dat in tqdm(val_dat[:]):
        code = dat["code"]
        main_method_name = dat["main_method_name"]
        allresults = dat["results"]

        samp = dict()
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

            samp['input'] = f'[INST] Please generate the data flow of the Jave code {code} from source {main_method_name} to sink {method_name} [\INST]'
            samp['output'] = f'<s>\t{dataflow}<\s>'
            
            newdat_val.append(samp)

        count_val += 1    

    for dat in tqdm(test_dat[:]):
        code = dat["code"]
        main_method_name = dat["main_method_name"]
        results = dat["results"]

        samp = dict()
        for result in results:
            method_name = list(result.keys())[0]
            if(method_name ==""):
                continue
            if(result[method_name] == []):
                continue
            data_flow_result = result[method_name][0]
            data_flow_result = '->'.join(data_flow_result)

            samp['input'] = f'[INST] Please generate the data flow of the Jave code {code} from source {main_method_name} to sink {method_name} [\INST]'
            samp['main_method_name'] = main_method_name
            samp['sink'] = method_name

            samp['output'] = f'<s>\t{data_flow_result}<\s>'
            
            newdat_test.append(samp)

        count_test += 1
    newdats_train = json.dumps(newdat_train, indent=2)
    newdats_val = json.dumps(newdat_val, indent=2)
    newdats_test = json.dumps(newdat_test, indent=2)
    print(f"training samples: {count_train}")
    print(f"val samples: {count_val}")
    print(f"test samples: {count_test}")
    print(f"count no data flow : {count_one}")
    with open("data/data_analysis_java_train.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_train)
    with open("data/data_analysis_java_test.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_test)
    with open("data/data_analysis_java_val.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_val)

