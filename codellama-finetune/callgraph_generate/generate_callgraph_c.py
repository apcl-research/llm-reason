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

output_filename = config['output_filename']
model_name_or_path = config['model_name_or_path']
adapter_path = config['adapter_path']
testfile = config["testfile"]




# TODO: Update variables
max_new_tokens = 500
top_p = 1
temperature = 1e-9


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, device_map='auto')

print("Loaded tokenizer")

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


count = 0
for data in tqdm.tqdm(testdata[:]):
    code = data["code"]
    method_name = data["function_name"]
    prompt = f'[INST] Given the C code {code}, Please generate the complete callgraph path of the method {method_name} [\INST]'
    ret = generate(model, prompt)
    ret = ret.split("[\INST]")[-1]
    ret = ret.split("<\s>")[0]
    data["result"] = ret.strip()
    predicted_results.append(data)
    if(count % 20 == 0):
        pickle.dump(predicted_results, open(output_filename, "wb"))
    count += 1
pickle.dump(predicted_results, open(output_filename, "wb"))
print(f"\033[91m Time: {time.time() - start}. \033[0m")
