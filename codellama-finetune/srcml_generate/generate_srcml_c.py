import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import time
import json
import tqdm
import pickle
import yaml


def read_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


config = read_config()

srcml_filename = config['srcml_filename']
testfile = config['testfile']
model_name_or_path = config["model_name_or_path"]
adapter_path = ["adapter_path"]




# TODO: Update variables
max_new_tokens = 500
top_p = 1
temperature = 1e-9



tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, device_map='auto')



model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    device_map = 'auto',
)
model.resize_token_embeddings(len(tokenizer)+1)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()


def generate(model, user_question, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer(user_question, return_tensors="pt").to('cuda')
    if(inputs['input_ids'].shape[1] > 1024):
        inputs['input_ids'] = inputs['input_ids'][:, -1024:]
        inputs['attention_mask'] = inputs['attention_mask'][:, -1024:]
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

start = time.time()
results = {}
count = 0 

testdata = pickle.load(open(testfile, "rb"))

predicted_results = []

example = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<unit xmlns="http://www.srcML.org/srcML/src" revision="1.0.0" language="C" filename="test.c"><function><type><name>int</name></type> <name>main</name><parameter_list>()</parameter_list> <block>{<block_content>    <expr_stmt><expr><call><name>printf</name><argument_list>(<argument><expr><literal type="string">"Hello, World!\n"</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <return>return <expr><literal type="number">0</literal></expr>;</return>
</block_content>}</block></function>
</unit>'''
example_code = '''
    int main() {
    printf("Hello, World!\n");
    return 0;
}
    '''

count = 0
for data in tqdm.tqdm(testdata[:]):
    code = data["code"]
    method_name = data["function_name"]
    prompt = f'[INST] Here\'s the example code {example_code} and the srcml of that code {example}. Given the code {code}, please generate the complete srcml of that code. Please remember to follow the example. Here\'s the srcml: [\INST]'
    ret = generate(model, prompt)
    ret = ret.split("[\INST]")[-1]
    ret = ret.split("<\s>")[0]
    data["result"] = ret.strip()
    predicted_results.append(data)
    if(count % 20 == 0):
        pickle.dump(predicted_results, open(srcml_filename, "wb"))
    count += 1
pickle.dump(predicted_results, open(srcml_filename, "wb"))
print(f"\033[91m Time: {time.time() - start}. \033[0m")
# import pdb; pdb.set_trace()
