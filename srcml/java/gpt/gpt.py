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

## prevent output from being blocked
client = OpenAI(api_key=key)
model_name = "gpt-4o-mini"




count = 0

srcmls = {}
fid_list = []
if os.path.exists(OUT_FILENAME):
    srcmls = pickle.load(open(OUT_FILENAME, "rb"))
number_of_functions = 0
for fid in srcmls:
    fid_list.append(fid)
    number_of_functions += 1

for fid in tqdm(q90_fids[:]):
    code = all_functions[fid]
    if(fid in fid_list):
        continue
    prompt = f"Given the code {code}, please generate the complete srcml of the given code without any explanation and markdown. Please remember to follow srcml rules"
    message = [{"role":"system", "content":"srcml generation"},
            {"role":"user", "content":prompt}]


    try:
        completion = client.chat.completions.create(
                model=model_name,
                messages=message
            )
        ret = completion.choices[0].message.content
        number_of_functions += 1
    except:
        continue

    srcmls[fid] = {"srcml":ret, "code": code}
    if(count % 5 == 0):
        pickle.dump(srcmls, open(f"{OUT_FILENAME}", "wb"))
    count += 1
    #if(number_of_functions >= 25000):
    #    break


pickle.dump(srcmls, open(f"{OUT_FILENAME}", "wb"))







