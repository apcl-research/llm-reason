from openai import OpenAI
import pickle
import os
from tqdm import tqdm
import re
import yaml


def read_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

# Read at the start of the script
config = read_config()

datafile = config['datafile']
OUT_FILENAME = config['OUT_FILENAME']


with open('./openai.key', 'r') as f:
    key = f.readline().strip()

client = OpenAI(api_key=key)
model_name = "gpt-4o-mini"



data_analysis_results = []

count = 0

for data in tqdm(alldata[:]):
    main_method_name = data["main_method_name"]
    source_code = data["code"]
    tools_results = data["results"]
    sinks = []
    gpt_result = {}
    for result in tools_results:
        sinks.extend(list(result.keys()))
    if(sinks ==[]):
        sinks.append("")
    gpt_result["main_method_name"] = main_method_name
    gpt_result["code"] = source_code
    gpt_result["results"] = []
    temp_results = {}
    for sink in sinks:

        prompt = f"Given the code {source_code}, please do the data analysis from source {main_method_name} to sink {sink} and print the statements, but without any explanation and markdown. Also, treat each statement as a statment instead of a block. Return the result with the template result:\t<statement1> -> <statement2>\n result:\t<statement1> -> <statement2>. Remember to use template as well."
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
        temp_results[sink] = ret
        


    gpt_result["results"].append(temp_results)
    data_analysis_results.append(gpt_result)
    if(count % 10 ==0):

        pickle.dump(data_analysis_results, open(OUT_FILENAME, "wb"))
    count += 1
pickle.dump(data_analysis_results, open(OUT_FILENAME, "wb"))
