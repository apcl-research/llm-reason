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



alldata = pickle.load(open(datafile, "rb"))

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
        example_code = '''public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);                                                                                                                                                                                                                                                                               // Source: Get input from the user                                                                                                                          System.out.println("Enter a number: ");
        int number = scanner.nextInt();                                                                                                                                                                                                                                                                                         // Pass data to the sink                                                                                                                                    processInput(number);                                                                                                                                                                                                                                                                                                   scanner.close();                                                                                                                                        }                                                                                                                                                                                                                                                                                                                       // Sink: Process the input data
    public static void processInput(int input) {
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

        prompt = f"Given the example code {example_code} and its data flow analysis result {example_data_flow} from source processInput to sink println . Given the code {source_code}, please do the data analysis from source {main_method_name} to sink {sink} and print the statements, but without any explanation and markdown. Please follow the example to generate the data flow analysis result."

        message = [{"role":"system", "content":"data flow analysis"},
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
