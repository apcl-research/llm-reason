import transformers
import torch
import subprocess
import pickle
import os
import re
from tqdm import tqdm
import networkx as nx
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import yaml

def read_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

# Read at the start of the script
config = read_config()


datafile = config['datafile']
OUT_FILENAME = config['OUT_FILENAME']
model_id = config['model_id']

model_id = "../llm_reason/CodeLlama-13b-Instruct-hf/"
OUT_FILENAME = "./data_flow_results_codellama_java_part3.pkl"
datafile = "/nfs/dropbox/llm_reason/data_analysis/data_flow_results_java_test.pkl"


alldata = pickle.load(open(datafile, "rb"))
data_analysis_results = []

count = 0
predicted_main_methods = []
if os.path.exists(OUT_FILENAME):
    data_analysis_results = pickle.load(open(OUT_FILENAME, "rb"))
for data in data_analysis_results:
    main_method  = data["main_method_name"]
    predicted_main_methods.append(main_method)

quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)


for data in tqdm(alldata[:]):
    main_method_name = data["main_method_name"]
    if(main_method_name in predicted_main_methods):
        continue
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
    example_code = '''public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);                                                                                                                                                                                                                                                                               // Source: Get input from the user                                                                                                                          System.out.println("Enter a number: ");
int number = scanner.nextInt();                                                                                                                                                                                                                                                                                         // Pass data to the sink                                                                                                                                    processInput(number);                                                                                                                                                                                                                                                                                                   scanner.close();                                                                                                                                        }                                                                                                                                                                                                                                                                                                                       // Sink: Process the input data                                                                                                                                                                                        public static void processInput(int input) {
        System.out.println("Processing the number: " + input);                                                                                                                                                                                                                                                                  if (input % 2 == 0) {                                                                                                                                           System.out.println(input + " is an even number.");
        } else {
            System.out.println(input + " is an odd number.");
        }                                                                                                                                                       }                                                                                                                                                       }
        '''
    example_data_flow = '''
        result: public static void processInput(int input) { ->  System.out.println("Processing the number: " + input);
        result: public static void processInput(int input) { -> if (input % 2 == 0) { ->  System.out.println(input + " is an even number.");
        result: public static void processInput(int input) { -> if (input % 2 == 0) { -> System.out.println(input + " is an odd number.");
        '''
    for sink in sinks:
        prompt = f"Given the example code {example_code} and its data flow analysis result {example_data_flow} from source processInput to sink println . Given the code {source_code}, please do the data analysis from source {main_method_name} to sink {sink} and print the statements, but without any explanation, markdown, and white space in the end. Please follow the example to generate the data flow analysis result. Here's the output:\n"

        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        if(input_ids["input_ids"].shape[1] > 10000):
            del input_ids
            continue
        with torch.no_grad():
            output = quantized_model.generate(**input_ids, max_new_tokens=256)
        #print(output)
            ret = tokenizer.decode(output[0], skip_special_tokens=True)
            ret = ret.split("Here's the output:")[-1]
            ret = ret.strip()

        del input_ids


        temp_results[sink] = ret
    gpt_result["results"].append(temp_results)
    data_analysis_results.append(gpt_result)
    if(count % 10 ==0):

        pickle.dump(data_analysis_results, open(OUT_FILENAME, "wb"))
    count += 1
pickle.dump(data_analysis_results, open(OUT_FILENAME, "wb"))

