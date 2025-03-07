from openai import OpenAI
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



key = ""
with open('./openai.key', 'r') as f:
    key = f.readline().strip()


client = OpenAI(api_key=key)
model_name = "gpt-4o-mini"




allcallgraphdata = pickle.load(open(callgraph_data_file, "rb"))
results = []

for data in tqdm(allcallgraphdata[:]):
    
    source_code = data["code"]
    function_name = data["method_name"]    

    example_code = '''
  static void myMethod() {
    System.out.println("Hello World!");
  }

  public static void main(String[] args) {
    myMethod();
  }
'''
    example_callgraph = '''
    path: main -> myMethod
    '''

    prompt = f"Given the example code {example_code} and example callgraph {example_callgraph} for the function main. Given the code {source_code} and the name of the method {function_name}, please generate all paths of the  callgraph for method {function_name} in the format path: a->b->c etc, where -> means call and a, b, c are just the name of the function. please just generate the graph without any explanation, markdown, additional information and remember the format is path: a->b->c"
    

    message = [{"role":"system", "content":"callgraph generation"},
            {"role":"user", "content":prompt}]

    
    try:
        completion = client.chat.completions.create(
                model=model_name,
                messages=message
            )
        ret = completion.choices[0].message.content
        data["gpt_result"] = ret
    except:
        continue
    results.append(data)
    if(len(results) % 5 == 0):
        pickle.dump(results, open(f"{OUT_FILENAME}", "wb"))
pickle.dump(results, open(f"{OUT_FILENAME}", "wb"))

