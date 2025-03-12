import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import time
import json
import tqdm
import yaml


def read_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


config = read_config()

output_filename = config['output_filename']
testfile = config['testfile']
model_name_or_path = config["model_name_or_path"]
adapter_path = ["adapter_path"]


# TODO: Update variables
max_new_tokens = 500
top_p = 1
temperature = 1e-9
f = open(testfile)
pf = open(output_filename, 'w')
testdata = json.load(f)


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, device_map='auto')


print("Loaded tokenizer")

# Load the model (use bf16 for faster inference)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    #load_in_8bit=True,
    device_map = 'auto',
     quantization_config=BitsAndBytesConfig(
         load_in_4bit=True,
         bnb_4bit_compute_dtype=torch.bfloat16,
         bnb_4bit_use_double_quant=True,
         bnb_4bit_quant_type='nf4',
     ),
     low_cpu_mem_usage=True
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
for data in tqdm.tqdm(testdata[:]):
    fid = data['fid']
    ret = generate(model, data['input'])
    ret = ret.split("[\INST]")[-1]
    s = f'{fid}\t{ret}\n'
    pf.write(s)
    pf.flush()
pf.close() 

print(f"\033[91m Time: {time.time() - start}. \033[0m")
# import pdb; pdb.set_trace()
