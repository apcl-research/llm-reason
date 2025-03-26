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
pickle.dump(results, open(f"{OUT_FILENAME}", "wb"))
