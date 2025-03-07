from rapidfuzz import fuzz
import pickle
from tqdm import tqdm
import yaml

def read_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = read_config()

srcml_tools_filename = config['srcml_tools_filename']
srcml_gpt_filename = config['srcml_gpt_filename']



srcml_tools = pickle.load(open(srcml_tools_filename, "rb"))
srcml_gpt = pickle.load(open(srcml_gpt_filename, "rb"))



def fuzz_path_similarity(path1, path2):
    similarity = fuzz.ratio(path1, path2) / 100  # Convert to 0-1 scale
    return similarity


total_levenshtein = 0

number_of_functions = 0


for fid in tqdm(list(srcml_gpt.keys())[:]):

    gpt_srcml  = srcml_gpt[fid]["srcml"]
    
    tool_srcml = srcml_tools[fid]


    total_levenshtein += fuzz_path_similarity(tool_srcml, gpt_srcml)


mean_levenshtein = total_levenshtein / len(srcml_gpt)
print(f"levenshtein distance: {mean_levenshtein}")
