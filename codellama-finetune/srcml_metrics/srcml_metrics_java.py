from rapidfuzz import fuzz
import pickle
from tqdm import tqdm
import yaml


def read_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


config = read_config()

tool_filename = config['tool_filename']
llm_filename = config['llm_filename']
testfid_filename = config['testfid_filename']

srcml_tools = pickle.load(open(tool_filename, "rb"))
srcml_gpt = pickle.load(open(llm_filename, "rb"))
fids = list(pickle.load(open(testfid_filename, "rb")).keys())



def fuzz_path_similarity(path1, path2):
    similarity = fuzz.ratio(path1, path2) / 100  # Convert to 0-1 scale
    return similarity


total_levenshtein = 0

number_of_functions = 0



for fid in tqdm(fids[:]):

    gpt_srcml  = srcml_gpt[fid]
    
    tool_srcml = srcml_tools[fid].split(f"{fid}.java\">")[-1]

    total_levenshtein += fuzz_path_similarity(tool_srcml, gpt_srcml)


mean_levenshtein = total_levenshtein / len(srcml_gpt)
print(f"levenshtein distance: {mean_levenshtein}")
