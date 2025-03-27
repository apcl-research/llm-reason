from rapidfuzz import fuzz
import pickle
from tqdm import tqdm
import yaml


def read_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

# Read at the start of the script
config = read_config()

prediction_file = config['prediction_file']


srcml_gpt = pickle.load(open(prediction_file, "rb"))




def fuzz_path_similarity(path1, path2):
    similarity = fuzz.ratio(path1, path2) / 100  # Convert to 0-1 scale
    return similarity


total_levenshtein = 0

number_of_functions = 0


for data in tqdm(srcml_gpt[:]):
    code = data["code"]
    gpt  = data["result"]
    tool = data["srcml"]
    total_levenshtein += fuzz_path_similarity(tool, gpt)


mean_levenshtein = total_levenshtein / len(srcml_gpt)
print(f"levenshtein distance: {mean_levenshtein}")
