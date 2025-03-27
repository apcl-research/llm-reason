import google.generativeai as genai
import pickle
from tqdm import tqdm
import os
import yaml


def read_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

# Read at the start of the script
config = read_config()

datafile = config['datafile']
OUT_FILENAME = config['OUT_FILENAME']


all_functions = pickle.load(open(datafile, "rb"))

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

srcmls = []
for data in tqdm(all_functions[:]):
    code = data["code"]
    srcml = data["srcml"]
    prompt = f"Given the code {code}, please generate the complete srcml of the given code without any explanationu and markdown. Please remember to follow srcml rules"

    message = [{"role":"system", "content":"srcml generation"},
        {"role":"user", "content":prompt}]

    answer = model.generate_content(prompt, safety_settings=safety_settings)
    ret = answer.text
    #except:
    #    continue


    ret = ret.split("```xml")[-1]
    ret = ret.split("```")[0]
    ret = ret.strip()


    srcmls.append({"llm_result":ret, "srcml":srcml, "code": code})
    if(count % 5 == 0):
        pickle.dump(srcmls, open(f"{OUT_FILENAME}", "wb"))
    count += 1


pickle.dump(srcmls, open(f"{OUT_FILENAME}", "wb"))









