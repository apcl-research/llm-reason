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
    
    example = '''
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<unit xmlns="http://www.srcML.org/srcML/src" revision="1.0.0" language="C" filename="test.c"><function><type><name>int</name></type> <name>main</name><parameter_list>()</parameter_list> <block>{<block_content>  <expr_stmt><expr><call><name>printf</name><argument_list>(<argument><expr><literal type="string">"Hello World!"</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
  <return>return <expr><literal type="number">0</literal></expr>;</return>
</block_content>}</block></function>
</unit>
        '''
    example_code = '''

        int main() {
            printf("Hello World!");
            return 0;
        }
    '''
    prompt = f"Here's the example code {example_code} and the srcml of that code {example}. Given the code {code}, please generate the complete srcml of that code. Please remember to follow the example.. Here's the srcml:"


    try:
        message = [{"role":"system", "content":"srcml generation"},
            {"role":"user", "content":prompt}]

        answer = model.generate_content(prompt, safety_settings=safety_settings)
        ret = answer.text
    except:
        continue


    ret = ret.split("```xml")[-1]
    ret = ret.split("```")[0]
    ret = ret.strip()


    srcmls.append({"llm_result":ret, "srcml":srcml, "code": code})
    if(count % 5 == 0):
        pickle.dump(srcmls, open(f"{OUT_FILENAME}", "wb"))
    count += 1

print(srcmls)
pickle.dump(srcmls, open(f"{OUT_FILENAME}", "wb"))









