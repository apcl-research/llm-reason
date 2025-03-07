import pickle
from tqdm import tqdm
import os
import subprocess
import transformers
import torch
import re
from tqdm import tqdm
import networkx as nx
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


import yaml

def read_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = read_config()

function_file_dir = config['function_file_dir']
q90_fid_file = config['q90_fid_file']
OUT_FILENAME = config['OUT_FILENAME']
model_id = config['model_id']




all_functions = pickle.load(open(function_file_dir, "rb"))
q90_fids = pickle.load(open(q90_fid_file, "rb"))

quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)



count = 0

srcmls = {}


for fid in tqdm(q90_fids[:]):
    code = all_functions[fid]
    srcml_list = []
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
    prompt = f"Here's the example code {example_code} and the srcml of that code {example}. Given the code {code}, please generate the complete srcml of that code. Please remember to follow the example.. Here's the srcml:"


    try:
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        if(input_ids["input_ids"].shape[1] > 10000):
            del input_ids
            continue
        with torch.no_grad():
            output = quantized_model.generate(**input_ids, max_new_tokens=2048)
        ret = tokenizer.decode(output[0], skip_special_tokens=True)
        srcml = ret.split("Here's the srcml:")[-1]
        number_of_functions += 1
    except:
        continue

    srcmls[fid] = {"srcml":srcml,  "code": code}
    if(count % 5 == 0):
        pickle.dump(srcmls, open(f"{OUT_FILENAME}", "wb"))
    count += 1


pickle.dump(srcmls, open(f"{OUT_FILENAME}", "wb"))







