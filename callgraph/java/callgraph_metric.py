import pickle
from rapidfuzz import fuzz
import re
import networkx as nx
from tqdm import tqdm

dat = pickle.load(open("/nfs/projects/llm_reason/callgraph/callgraph_java_gpt_test_q90_incontext.pkl", "rb"))

def fuzz_path_similarity(path1, path2):
    similarity = fuzz.ratio(path1, path2) / 100  # Convert to 0-1 scale
    return similarity

total_similarity = 0

for info in tqdm(dat[:]):
    true_paths = info["path"]
    true_callgraph = [[item.strip('"') for item in sublist] for sublist in true_paths]
    gpt_paths = info["gpt_result"]
    gpt_paths = re.findall(r'path:\s*(.*)', gpt_paths)
    gpt_callgraph = []
    for path in gpt_paths:
        path = path.split("->")
        path = [tmp.strip() for tmp in path]
        gpt_callgraph.append(path)
    gpt_callgraph_edges = []
    true_callgraph_edges = []
    for sequence in gpt_callgraph:
        edges = [(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
        gpt_callgraph_edges.extend(edges)
    
    for sequence in true_callgraph:
        edges = [(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
        true_callgraph_edges.extend(edges)
    
    

    truegraph = nx.DiGraph()
    truegraph.add_edges_from(true_callgraph_edges)


    gptgraph = nx.DiGraph()
    gptgraph.add_edges_from(gpt_callgraph_edges)

    
    edges_true = set(truegraph.edges())
    edges_gpt = set(gptgraph.edges())
    
    intersection = len(edges_true & edges_gpt)
    union = len(edges_true | edges_gpt)
    jaccard_similarity = intersection / union if union != 0 else 0
    
    total_similarity += jaccard_similarity
    
mean_jaccard_similarity =  total_similarity / len(dat)
print(f"mean jaccard similarity: {mean_jaccard_similarity}")

# pair accuracy 

total_acc = 0

for info in tqdm(dat[:]):
    true_paths = info["path"]
    true_callgraph = [[item.strip('"') for item in sublist] for sublist in true_paths]
    gpt_paths = info["gpt_result"]
    gpt_paths = re.findall(r'path:\s*(.*)', gpt_paths)
    gpt_callgraph = []
    for path in gpt_paths:
        path = path.split("->")
        path = [tmp.strip() for tmp in path]
        gpt_callgraph.append(path)
    gpt_callgraph_edges = []
    true_callgraph_edges = []
    for sequence in gpt_callgraph:
        edges = [(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
        gpt_callgraph_edges.extend(edges)
    
    for sequence in true_callgraph:
        edges = [(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
        true_callgraph_edges.extend(edges)
    
    truegraph = nx.DiGraph()
    truegraph.add_edges_from(true_callgraph_edges)


    gptgraph = nx.DiGraph()
    gptgraph.add_edges_from(gpt_callgraph_edges)

    
    edges_true = list(truegraph.edges())
    edges_gpt = list(gptgraph.edges())
    intersect_edges = list(set(edges_gpt).intersection(edges_true))
    number_of_intersect_edges = len(intersect_edges)
    if(edges_gpt != []):
        total_acc += number_of_intersect_edges / len(edges_gpt)
    elif(edges_gpt == [] and edges_true ==[]):
        total_acc += 1

mean_pair_accuracy = total_acc / len(dat)
print(f"mean pair accuracy: {mean_pair_accuracy}")




total_chain_acc = 0


for info in dat[:]:
    true_edges = info["path"]
    gpt_paths = info["gpt_result"]
    totla_method_score = 0
    
    true_paths = []

    gpt_paths = re.findall(r'path:\s*(.*)', gpt_paths)
    for edge in true_edges:
        edge = [e.replace("\"", "") for e in edge]
        true_paths.append('->'.join(edge).strip())
    for gpt_path in gpt_paths:
        if(gpt_path.strip() in true_paths):
            total_chain_acc += 1
    if(len(gpt_paths) != 0):
        total_chain_acc /= len(gpt_paths)
    elif(gpt_paths == [] and true_paths == []):
        total_chain_acc += 1
mean_chain_accuracy = total_chain_acc / len(dat)
print(f"mean chain accuracy: {mean_chain_accuracy}")



