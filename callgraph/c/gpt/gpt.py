from openai import OpenAI
import pickle
import os
from tqdm import tqdm
import re
import networkx as nx
import yaml


def read_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


config = read_config()

source_file = config['source_file']
OUT_FILENAME = config['OUT_FILENAME']




with open('openai.key', 'r') as f:
    key = f.readline().strip()

client = OpenAI(api_key=key)
model_name = "gpt-4o-mini"


results = []

alldata = pickle.load(open(source_file, "rb"))


for data in tqdm(alldata[:]):
    newdata = data.copy()
    source_code = newdata["code"] 
    function_name = newdata["function_name"]
    prompt = f"Given the code {source_code} and the name of the method {function_name}, please generate all paths of the  callgraph for method {function_name} in the format path: a->b->c etc, where -> means call and a, b, c are just the name of the function. please just generate the graph without any explanation, markdown, additional information and remember the format is path: a->b->c"

    message = [{"role":"system", "content":"callgraph generation"},
            {"role":"user", "content":prompt}]
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=message
        )
    except:
        continue
    
    ret = completion.choices[0].message.content
    newdata["results"] = ret
    results.append(newdata)
    if(len(results) % 5 == 0):
        pickle.dump(results, open(f"{OUT_FILENAME}", "wb"))
    print(f"current number of functions: {len(results)}")
    if(len(results) >= 25000):
        break
pickle.dump(results, open(f"{OUT_FILENAME}", "wb"))
