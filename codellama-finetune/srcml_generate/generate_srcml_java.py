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
f = open(testfile)
testdata = json.load(f)
# Base model
# Adapter name on HF hub or local checkpoint path.



tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, device_map='auto')



# Load the model (use bf16 for faster inference)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    device_map = 'auto',
)
model.resize_token_embeddings(len(tokenizer)+1)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()


prompt = "<s>[INST] {user_question}[\INST]"

def generate(model, user_question, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to('cuda')
    # inputs = tokenizer(user_question[0], return_tensors="pt").to('cuda')
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
for data in tqdm.tqdm(testdata[:]):
    fid = data['fid']
    ret = generate(model, data['input'])
    ret = ret.split("[\INST]")[-1]
    results[fid] = ret.strip()
    if(count % 20 == 0):
        pickle.dump(results, open(srcml_filename, "wb"))
    count += 1
pickle.dump(results, open(srcml_filename, "wb"))

print(f"\033[91m Time: {time.time() - start}. \033[0m")
