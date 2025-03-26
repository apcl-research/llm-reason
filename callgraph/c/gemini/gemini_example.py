import google.generativeai as genai
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

source_file = config['source_file']
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




results = []

alldata = pickle.load(open(source_file, "rb"))

for data in tqdm(alldata[:]):
    newdata = data.copy()
    source_code = newdata["code"].strip()
    function_name = newdata["function_name"]
    example_code = '''
        /**
     * @file example.c
 * @brief Example C program with Doxygen caller graph.
 */

/**
 * @brief Adds two integers.
 * @param a First integer.
 * @param b Second integer.
 * @return Sum of a and b.
 */
int add(int a, int b) {
    return a + b;
}

/**
 * @brief Multiplies two integers.
 * @param a First integer.
 * @param b Second integer.
 * @return Product of a and b.
 */
int multiply(int a, int b) {
    return a * b;
}

/**
 * @brief Performs a series of calculations.
 * Calls add() and multiply() functions.
 * @param x First integer.
 * @param y Second integer.
 * @return Result of combined calculations.
 */
int calculate(int x, int y) {
    int sum = add(x, y);
    int product = multiply(x, y);
    return sum + product;
}

/**
 * @brief Main function.
 * Calls calculate() function.
 * @return Exit status.
 */
int main() {
    int result = calculate(3, 4);     
     return result;
}
    '''
    example_callgraph = '''
    path: calculate -> add
    path: calculate -> multiply
    '''
    prompt = f"Given the example code {example_code}  and example callgraph {example_callgraph} for the function calculate. Given the code {source_code}, please follow the example to generate all paths of the  callgraph for method {function_name} in the format path: a->b->c etc, where -> means call. please just generate the graph without any explanation, markdown, additional information and remember the format is path: a->b->c"


    try:
        answer = model.generate_content(prompt, safety_settings=safety_settings)
        ret = answer.text
    except:
        continue
    newdata["results"] = ret
    results.append(newdata)
    if(len(results) % 5 == 0):
        pickle.dump(results, open(f"{OUT_FILENAME}", "wb"))
    time.sleep(1)
pickle.dump(results, open(f"{OUT_FILENAME}", "wb"))
