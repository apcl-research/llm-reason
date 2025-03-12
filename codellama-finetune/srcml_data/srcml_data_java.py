import json
import argparse
import pickle
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train-filename', type=str, default='/nfs/projects/llm_reason/srcml/train/srcml_java.pkl')
    parser.add_argument('--test-filename', type=str, default='/scratch/chiayi/llm_reason/srcml/srcml_result_codellama_java_function_test.pkl')

    args = parser.parse_args()
    train_filename = args.train_filename
    test_filename = args.test_filename

    newdat_train = list()
    newdat_val = list()
    newdat_test = list()
    
    train_val_data = pickle.load(open(train_filename, "rb"))
    testdata = pickle.load(open(test_filename, "rb"))
    
    train_data = list(train_val_data.keys())[:25000]
    val_data = list(train_val_data.keys())[25000:30000]


    count_train = 0
    count_val = 0
    count_test = 0
    
    example = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<unit xmlns="http://www.srcML.org/srcML/src" revision="1.0.0" language="Java" filename="test.java"><function><type><specifier>public</specifier> <specifier>static</specifier> <name>void</name></type> <name>main</name><parameter_list>(<parameter><decl><type><name><name>String</name><index>[]</index></name></type> <name>args</name></decl></parameter>)</parameter_list> <block>{<block_content>
    <expr_stmt><expr><call><name><name>System</name><operator>.</operator><name>out</name><operator>.</operator><name>println</name></name><argument_list>(<argument><expr><literal type="string">"Hello World"</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
  </block_content>}</block></function>
</unit>'''
    example_code = '''
    public static void main(String[] args) {
    System.out.println("Hello World");
  }
  '''



    for fid in tqdm(train_data[:]):
        code = train_val_data[fid]["code"]
        srcml = train_val_data[fid]["srcml"]
        samp = {}
        samp['fid'] = count_train
        samp['input'] = f"Here's the example code {example_code} and the srcml of that code {example}. Given the code {code}, please generate the complete srcml of that code. Please remember to follow the example. Here's the srcml:"
        samp['output'] = f'<s>\t{srcml}</s>'
        newdat_train.append(samp)
        count_train += 1

    for fid in tqdm(val_data[:]):
        code = train_val_data[fid]["code"]
        srcml = train_val_data[fid]["srcml"]
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


    
    newdats_train = json.dumps(newdat_train, indent=2)
    newdats_val = json.dumps(newdat_val, indent=2)
    newdats_test = json.dumps(newdat_test, indent=2)
    print(f"training samples: {count_train}")
    print(f"val samples: {count_val}")
    print(f"test samples: {count_test}")
    with open("data/srcml_java_train.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_train)
    with open("data/srcml_java_test.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_test)
    with open("data/srcml_java_val.json", 'w', encoding='utf8') as json_file:
        json_file.write(newdats_val)
