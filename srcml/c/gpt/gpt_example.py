import pickle
from openai import OpenAI
from tqdm import tqdm
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

with open('openai.key', 'r') as f:
    key = f.readline().strip()

client = OpenAI(api_key=key)
model_name = "gpt-4o-mini"

count = 0

results = []

for data in tqdm(all_functions[:]):
    code = data["code"]
    srcml = data["srcml"]
    
    example = '''
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<unit xmlns="http://www.srcML.org/srcML/src" revision="1.0.0" language="C" filename="test.c"><function><type><name>int</name></type> <name>main</name><parameter_list>()</parameter_list> <block>{<block_content>
  <expr_stmt><expr><call><name>printf</name><argument_list>(<argument><expr><literal type="string">"Hello World!"</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
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



    message = [{"role":"system", "content":"srcml generation"},
        {"role":"user", "content":prompt}]
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=message
        )
    except:
        continue

    ret = completion.choices[0].message.content
    ret = ret.split("```xml")[-1]
    ret = ret.split("```")[0]
    results.append({"llm_result":ret,  "code": code, "srcml":srcml})
    if(count % 5 == 0):
        pickle.dump(results, open(f"{OUT_FILENAME}", "wb"))
    count += 1
    
pickle.dump(results, open(f"{OUT_FILENAME}", "wb"))
