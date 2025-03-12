import json
import argparse
import pickle
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train-filename', type=str, default='/nfs/projects/llm_reason/c_code/functions_new_train_srcml.pkl')
    parser.add_argument('--test-filename', type=str, default='/scratch/chiayi/llm_reason/srcml/srcml_result_codellama_java_function_test.pkl')
    #parser.add_argument('--com-filename', type=str, default='/nfs/projects/funcom/data/javastmt_fc/output/coms.val')

    args = parser.parse_args()
    train_filename = args.train_filename
    test_filename = args.test_filename

    newdat_train = list()
    newdat_val = list()
    newdat_test = list()
    
    train_val_data = pickle.load(open(train_filename, "rb"))
    testdata = pickle.load(open(test_filename, "rb"))
    
    train_data = train_val_data[:10000]
    val_data = train_val_data[10000:]


    count_train = 0
    count_val = 0
    count_test = 0
    
    example = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<unit xmlns="http://www.srcML.org/srcML/src" revision="1.0.0" language="C" filename="test.c"><function><type><name>int</name></type> <name>main</name><parameter_list>()</parameter_list> <block>{<block_content>
    <expr_stmt><expr><call><name>printf</name><argument_list>(<argument><expr><literal type="string">"Hello, World!\n"</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <return>return <expr><literal type="number">0</literal></expr>;</return>
</block_content>}</block></function>
</unit>'''
    example_code = '''
    int main() {
    printf("Hello, World!\n");
    return 0;
}
    '''



    for data in tqdm(train_data[:]):
        code = data["code"]
        srcml = data["srcml"]
        samp = {}
        samp['fid'] = count_train
        samp['input'] = f"Here's the example code {example_code} and the srcml of that code {example}. Given the code {code}, please generate the complete srcml of that code. Please remember to follow the example. Here's the srcml:"
        samp['output'] = f'<s>\t{srcml}</s>'
        newdat_train.append(samp)
        count_train += 1

    for data in tqdm(val_data[:]):
        code = data["code"]
        srcml = data["srcml"]
        samp = {}
        samp['fid'] = count_val
        samp['input'] = f"Here's the example code {example_code} and the srcml of that code {example}. Given the code {code}, please generate the complete srcml of that code. Please remember to follow the example. Here's the srcml:"
        samp['output'] = f'<s>\t{srcml}</s>'
        newdat_val.append(samp)
        count_val += 1
    
    for fid in tqdm(list(testdata.keys())[:]):
        code = testdata[fid]["code"]
        srcml = testdata[fid]["srcml"]
        samp = {}
        samp['fid'] = fid 
        samp['input'] = f"Here's the example code {example_code} and the srcml of that code {example}. Given the code {code}, please generate the complete srcml of that code. Please remember to follow the example. Here's the srcml:"
        samp['output'] = f'<s>\t'
        newdat_test.append(samp)
        count_test += 1


    
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
    with open("data/srcml_c_train.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_train)
    with open("data/srcml_c_test.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_test)
    with open("data/srcml_c_val.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_val)
