import transformers
import torch
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import pickle
from tqdm import tqdm
import time
import os
import re
import networkx as nx
import yaml

def read_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = read_config()

callgraph_data_file = config['callgraph_data_file']
OUT_FILENAME = config['OUT_FILENAME']
model_id = config['model_id']




all_callgraph_data = pickle.load(open(callgraph_data_file, "rb"))
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
results = []

for data in tqdm(all_callgraph_data[:]):
    source_code = data["code"]
    function_name = data["method_name"]

    prompt = f"Given the code {source_code} and the name of the method {function_name}, please generate all paths of the  callgraph for method {function_name} in the format path: a->b->c etc, where -> means call and a, b, c are just the name of the function. please just generate the graph without any explanation, markdown, additional information and remember the format is path: a->b->c"
    
    
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    if(input_ids["input_ids"].shape[1] > 10000):
        del input_ids

        continue
    with torch.no_grad():
        output = quantized_model.generate(**input_ids, max_new_tokens=256)
        ret = tokenizer.decode(output[0], skip_special_tokens=True)
        ret = ret.split("Answer: \\begin{code}")[-1]
        ret = ret.split("\end{code}")[0]
        ret = ret.strip()
        data["llm_result"] = ret
    del input_ids
    
    results.append(data)
    if(len(results) % 5 == 0):
        pickle.dump(results, open(f"{OUT_FILENAME}", "wb"))
pickle.dump(results, open(f"{OUT_FILENAME}", "wb"))
