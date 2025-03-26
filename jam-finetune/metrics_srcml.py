import pickle
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import javalang
import diff_match_patch as dmp_module
import statistics
from rapidfuzz import fuzz

def fuzz_path_similarity(path1, path2):
    similarity = fuzz.ratio(path1, path2) / 100  # Convert to 0-1 scale
    return similarity

srcml_dir = 'srcml_prediction_new'
q90testsrcmlfile = '/nublar/datasets/jm52m/q90testfids_srcml.pkl'
q90testfidsfile = '/nublar/datasets/jm52m/q90testfids.pkl'


exec(open('configurator.py').read()) # overrides from command line or config file





q90testsrcml = pickle.load(open(q90testsrcmlfile, 'rb'))
q90testfids = pickle.load(open(q90testfidsfile, 'rb'))

total = 0

for fid in tqdm(q90testfids[:]):
  
  #print(f'{fid}\t', end='', flush=True)
  
    try:
        with open(f'{srcml_dir}/{fid}.xml', 'r', encoding='utf-8') as f:
            srcml_pred = f.read()
    except:
        #print('text')
        continue


    srcml_ref = q90testsrcml[fid]
    srcml_ref = srcml_ref.strip()
    srcml_pred = srcml_pred.strip()
    
    
  
    srcml_edit_distance = fuzz_path_similarity(srcml_ref, srcml_pred)
    total += srcml_edit_distance
print(f"levenshtein: {total/len(q90testfids)}")


