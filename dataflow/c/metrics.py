import pickle
import re
import networkx as nx
from tqdm import tqdm



tools_results = pickle.load(open("/nfs/projects/llm_reason/data_analysis/c/data_flow_results_c_new_test_filtered.pkl", "rb"))
llm_results = pickle.load(open("/home/chiayi/project/qlora-codellama/predictions/dataflow_c_codellama.pkl", "rb"))

tool_main_methods = []
llm_main_methods = [] 


for result in tools_results:
    main_method_name = result["main_method_name"]
    tool_main_methods.append(main_method_name)

for result in llm_results:
    main_method_name = result["main_method_name"]
    llm_main_methods.append(main_method_name)

total_similarity = 0
count_total_error = 0
for main_method_name in tqdm(tool_main_methods[:]):
    try:
        llm_result_index = llm_main_methods.index(main_method_name)
    except:
        continue
    tool_result_index = tool_main_methods.index(main_method_name)
    tool_result = tools_results[tool_result_index]
    llm_result = llm_results[llm_result_index]
    tool_data_flow = tool_result["results"]
    llm_data_flow = llm_result["results"][0]
    
    temp_score = 0
    count_error = 0
    for alldata in tool_data_flow[:]:
        for sink in list(alldata.keys()):
            tool_dataflow_graph = nx.DiGraph()
            llm_dataflow_graph = nx.DiGraph()
            tool_paths = alldata[sink]
            for tool_path in tool_paths[:]:
                tool_edges = [(tool_path[i], tool_path[i + 1]) for i in range(len(tool_path) - 1)]
                tool_dataflow_graph.add_edges_from(tool_edges)
                try:
                    llm_path = llm_data_flow[sink]
                except:
                    count_total_error += 1
                    count_error += 1
                    continue
                all_llm_path = llm_path.split("\n") 
                llm_paths = []
                for path in all_llm_path:
                    path = path.split("result:")[-1]
                    path = path.split("->")
                    path = [tmp.strip() for tmp in path]
                    llm_paths.append(path)
                for llm_path in llm_paths:
                    llm_path =  [s for s in llm_path if s] 
                    llm_edges = [(llm_path[i], llm_path[i + 1]) for i in range(len(llm_path) - 1)]
                    llm_dataflow_graph.add_edges_from(llm_edges)

                llm_edges = set(llm_dataflow_graph.edges())
                tool_edges = set(tool_dataflow_graph.edges())

                intersection = len(llm_edges & tool_edges)
                union = len(llm_edges | tool_edges)
                jaccard_similarity = intersection / union if union != 0 else 0
                temp_score += jaccard_similarity
    temp_score = temp_score / (len(tool_data_flow) + len(list(alldata.keys())) + len(tool_paths) + len(llm_paths) - count_error)
    total_similarity += temp_score
                #total_similarity += jaccard_similarity
            #print(paths)
mean_jaccard_similarity =  total_similarity / len(tool_main_methods)
print(f"mean jaccard similarity: {mean_jaccard_similarity}")
print(f"number of error methods: {count_total_error}")



total_acc = 0

for main_method_name in tool_main_methods[:]:
    try:
        llm_result_index = llm_main_methods.index(main_method_name)
    except:
        continue
    tool_result_index = tool_main_methods.index(main_method_name)
    tool_result = tools_results[tool_result_index]
    llm_result = llm_results[llm_result_index]
    tool_data_flow = tool_result["results"]
    llm_data_flow = llm_result["results"][0]
    #for 
    temp_score = 0
    count_error = 0
    for alldata in tool_data_flow:
        for sink in list(alldata.keys()):
            tool_dataflow_graph = nx.DiGraph()
            llm_dataflow_graph = nx.DiGraph()
            tool_paths = alldata[sink]
            #print("-----", sink)
            for tool_path in tool_paths[:]:
                tool_edges = [(tool_path[i], tool_path[i + 1]) for i in range(len(tool_path) - 1)]
                tool_dataflow_graph.add_edges_from(tool_edges)
                try:
                    llm_path = llm_data_flow[sink]
                except:
                    count_error += 1
                    continue
                all_llm_path = llm_path.split("\n")
                llm_paths = []
                for path in all_llm_path:
                    path = path.split("result:")[-1]
                    path = path.split("->")
                    path = [tmp.strip() for tmp in path]
                    llm_paths.append(path)
                for llm_path in llm_paths:
                    llm_path =  [s for s in llm_path if s]
                    llm_edges = [(llm_path[i], llm_path[i + 1]) for i in range(len(llm_path) - 1)]
                    llm_dataflow_graph.add_edges_from(llm_edges)

                edges_llm = list(llm_dataflow_graph.edges())
                edges_tools = list(tool_dataflow_graph.edges())
                intersect_edges = list(set(edges_tools).intersection(edges_llm))
                number_of_intersect_edges = len(intersect_edges)
                if(edges_llm != []):
                    temp_score = temp_score + (number_of_intersect_edges / len(edges_llm))
                elif(edges_tools == [] and edges_llm ==[]):
                    temp_score += 1
    total_acc  += temp_score / (len(tool_data_flow) + len(list(alldata.keys())) + len(tool_paths) - count_error)
mean_pair_accuracy = total_acc / len(tool_main_methods)
print(f"mean pair accuracy: {mean_pair_accuracy}")


total_chain_acc = 0 

for main_method_name in tool_main_methods[:]:
    try:
        llm_result_index = llm_main_methods.index(main_method_name)
    except:
        continue
    tool_result_index = tool_main_methods.index(main_method_name)
    tool_result = tools_results[tool_result_index]
    llm_result = llm_results[llm_result_index]
    tool_data_flow = tool_result["results"]
    llm_data_flow = llm_result["results"][0]
    #for 
    temp_score = 0
    count_error = 0
    for alldata in tool_data_flow:
        for sink in list(alldata.keys()):
            tool_dataflow_graph = nx.DiGraph()
            llm_dataflow_graph = nx.DiGraph()
            tool_paths = alldata[sink]
            #print("-----", sink)
            for tool_path in tool_paths[:]:
                tool_edges = [(tool_path[i], tool_path[i + 1]) for i in range(len(tool_path) - 1)]
                tool_dataflow_graph.add_edges_from(tool_edges)
                try:
                    llm_path = llm_data_flow[sink]
                except:
                    count_error += 1
                    continue
                
                all_llm_path = llm_path.split("\n")
                llm_paths = []
                for path in all_llm_path:
                    path = path.split("result:")[-1]
                    path = path.split("->")
                    path = [tmp.strip() for tmp in path]
                    llm_paths.append(path)
                for llm_path in llm_paths:
                    llm_path =  [s for s in llm_path if s]
                    llm_edges = [(llm_path[i], llm_path[i + 1]) for i in range(len(llm_path) - 1)]
                    llm_dataflow_graph.add_edges_from(llm_edges)


                edges_llm = list(llm_dataflow_graph.edges())
                edges_tools = list(tool_dataflow_graph.edges())
                
                if(edges_llm == edges_tools):
                    temp_score += 1
    total_chain_acc  += temp_score / (len(tool_data_flow) + len(list(alldata.keys())) + len(tool_paths) - count_error)
mean_chain_accuracy = total_chain_acc / len(tool_main_methods)
print(f"mean chain accuracy: {mean_chain_accuracy}")






