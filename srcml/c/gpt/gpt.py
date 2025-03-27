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
    prompt = f"Given the code {code}, please generate the complete srcml of the given code without any explanationu and markdown. Please remember to follow srcml rules"

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
    results.append({"llm_result":ret,  "code": code, "srcml":srcml})
    if(count % 5 == 0):
        pickle.dump(results, open(f"{OUT_FILENAME}", "wb"))
    count += 1
    
pickle.dump(results, open(f"{OUT_FILENAME}", "wb"))
