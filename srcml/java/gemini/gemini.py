import google.generativeai as genai
import pickle
from tqdm import tqdm
import os
import yaml

def read_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


function_file_dir = config['function_file_dir']
q90_fid_file = config['q90_fid_file']
OUT_FILENAME = config['OUT_FILENAME']






q90_fids = pickle.load(open(q90_fid_file, "rb"))
all_functions = pickle.load(open(function_file_dir, "rb"))

## prevent output from being blocked
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


with open('gemini.key', 'r') as f:
    key = f.readline().strip()

genai.configure(api_key=key)

model = genai.GenerativeModel('gemini-1.5-flash')

count = 0

srcmls = {}

for fid in tqdm(q90_fids[:]):
    code = all_functions[fid]
    prompt = f"Given the code {code}, please generate the complete srcml of the given code without any explanation and markdown. Please remember to follow srcml rules"

    message = [{"role":"system", "content":"srcml generation"},
            {"role":"user", "content":prompt}]

    try:
        answer = model.generate_content(prompt, safety_settings=safety_settings)
        ret = answer.text
        number_of_functions += 1
    except:
        continue

    srcmls[fid] = {"srcml":ret, "code": code}
    if(count % 5 == 0):
        pickle.dump(srcmls, open(f"{OUT_FILENAME}", "wb"))
    count += 1


pickle.dump(srcmls, open(f"{OUT_FILENAME}", "wb"))







