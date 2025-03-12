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
testfile = config['testfile']
model_name_or_path = config["model_name_or_path"]
adapter_path = ["adapter_path"]


# TODO: Update variables
max_new_tokens = 500
top_p = 1
temperature = 1e-9


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, device_map='auto')

print("Loaded tokenizer")

# Load the model (use bf16 for faster inference)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    #device_map={"": 0},
    load_in_8bit=True,
    device_map = 'auto',
    )
model.resize_token_embeddings(len(tokenizer)+1)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()


prompt = "<s>[INST] {user_question}[\INST]"

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
allcode = []
predicted_data = []

count = 0
for data in tqdm.tqdm(testdata[:]):
    code = data["code"]
    main_method_name = data["main_method_name"]
    tools_results = data["results"]
    gpt_result = {}
    sinks = []
    for result in tools_results:
        sinks.extend(list(result.keys()))
    if(sinks ==[]):
        sinks.append("")
    gpt_result["main_method_name"] = main_method_name
    gpt_result["code"] = code
    gpt_result["results"] = []
    temp_results = {}
    for sink in sinks:
        prompt = f'[INST] Please generate the data flow of the Jave code {code} from source {main_method_name} to sink {sink} [\INST]' 
        ret = generate(model, prompt)
        ret = ret.split("[\INST]")[-1]
        ret = ret.split("<\s>")[0]
        temp_results[sink] = ret.strip()
    gpt_result["results"].append(temp_results)
    
    predicted_data.append(gpt_result)

    if(count % 20 == 0):
        pickle.dump(predicted_data, open(output_filename, "wb"))
    count += 1
pickle.dump(predicted_data, open(output_filename, "wb"))
print(f"\033[91m Time: {time.time() - start}. \033[0m")
# import pdb; pdb.set_trace()
