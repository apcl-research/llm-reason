from rapidfuzz import fuzz
import pickle
from tqdm import tqdm



srcml_tools = pickle.load(open("/nublar/datasets/jm52m/q90testfids_srcml.pkl", "rb"))
srcml_gpt = pickle.load(open("/home/chiayi/project/qlora-codellama/predictions/srcml_codellama.pkl", "rb"))
fids = list(pickle.load(open("/scratch/chiayi/llm_reason/srcml/srcml_result_codellama_java_function_test.pkl", "rb")).keys())
# /scratch/chiayi/llm_reason/srcml/srcml_result_gemini_c_function.pkl -- gemini
# /scratch/chiayi/llm_reason/srcml/srcml_result_gpt_mini_c_function.pkl -- gpt



def fuzz_path_similarity(path1, path2):
    similarity = fuzz.ratio(path1, path2) / 100  # Convert to 0-1 scale
    return similarity


total_levenshtein = 0

number_of_functions = 0



for fid in tqdm(fids[:]):

    gpt_srcml  = srcml_gpt[fid]
    
    tool_srcml = srcml_tools[fid].split(f"{fid}.java\">")[-1]
    #print("-----", gpt_srcml)
    #print(tool_srcml)

    total_levenshtein += fuzz_path_similarity(tool_srcml, gpt_srcml)


mean_levenshtein = total_levenshtein / len(srcml_gpt)
print(f"levenshtein distance: {mean_levenshtein}")
