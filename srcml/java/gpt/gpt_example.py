import pickle
from openai import OpenAI
from tqdm import tqdm
import os
import yaml

with open('./openai.key', 'r') as f:
    key = f.readline().strip()

def read_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = read_config()

function_file_dir = config['function_file_dir']
q90_fid_file = config['q90_fid_file']
OUT_FILENAME = config['OUT_FILENAME']



q90_fids = pickle.load(open(q90_fid_file, "rb"))
all_functions = pickle.load(open(function_file_dir, "rb"))

client = OpenAI(api_key=key)
model_name = "gpt-4o-mini"




count = 0

srcmls = {}
fid_list = []

for fid in tqdm(q90_fids[:]):
    code = all_functions[fid]
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
    prompt = f"Here's the example code {example_code} and the srcml of that code {example}. Given the code {code}, please generate the complete srcml of the given code without any explanation and markdown. Please remember to follow the given example"
    message = [{"role":"system", "content":"srcml generation"},
            {"role":"user", "content":prompt}]


    try:
        completion = client.chat.completions.create(
                model=model_name,
                messages=message
            )
        ret = completion.choices[0].message.content
    except:
        continue

    srcmls[fid] = {"srcml":ret, "code": code}
    if(count % 5 == 0):
        pickle.dump(srcmls, open(f"{OUT_FILENAME}", "wb"))
    count += 1


pickle.dump(srcmls, open(f"{OUT_FILENAME}", "wb"))






