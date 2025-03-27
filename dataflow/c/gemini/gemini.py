import pickle
import google.generativeai as genai
from tqdm import tqdm
import time
import os
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


key = ""
with open('gemini.key', 'r') as f:
    key = f.readline().strip()

genai.configure(api_key=key)

model = genai.GenerativeModel('gemini-1.5-flash')


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

alldata = pickle.load(open(datafile, "rb"))


data_analysis_results = []

count = 0

for data in tqdm(alldata[:]):
    main_method_name = data["main_method_name"]
    source_code = data["code"]
    tools_results = data["results"]
    sinks = []
    gemini_result = {}
    for result in tools_results:
        sinks.extend(list(result.keys()))
    if(sinks ==[]):
        sinks.append("")
    gemini_result["main_method_name"] = main_method_name
    gemini_result["code"] = source_code
    gemini_result["results"] = []
    temp_results = {}
    for sink in sinks:
        prompt = f"Given the code {source_code}, please do the data analysis from source {main_method_name} to sink {sink} and print the statements, but without any explanation and markdown. Also, treat each statement as a statment instead of a block. Return the result with the template result:\t<statement1> -> <statement2>\n result:\t<statement1> -> <statement2>. Remember to use template as well."
        try:
            answer = model.generate_content(prompt, safety_settings=safety_settings)
            ret = answer.text
            temp_results[sink] = ret

        except:
            continue
    gemini_result["results"].append(temp_results)
    data_analysis_results.append(gemini_result)
    if(count % 10 ==0):
        
        pickle.dump(data_analysis_results, open(OUT_FILENAME, "wb"))
    count += 1
pickle.dump(data_analysis_results, open(OUT_FILENAME, "wb"))
