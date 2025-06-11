import sys
import pickle, os, sys, re
import networkx as nx
import numpy as np
import pandas as pd
import gc 
sys.path.append("scripts")
sys.path.append("../../scripts")

from graph import OverlapGraph,remove_false_edges,get_neighbor_overlap_bases

MAX_SAMPLE_SIZE = int(1e9)
COVERAGE_DEPTH = 20

sample = snakemake.wildcards['sample']
dataset = snakemake.wildcards['platform']
region = snakemake.wildcards['region']
method = snakemake.wildcards['method']

ref_graph_path = snakemake.input['ref_graph']
tsv_path = snakemake.input['metadata']
nbr_path = snakemake.input['nbr_indice']
df_file= snakemake.output['integral_stat']

filename = os.path.abspath(nbr_path)
pattern  = r'data/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/(.+)_nbr_matrix'
thread = re.search(pattern,filename).group(1)
sample = re.search(pattern,filename).group(2)
region = re.search(pattern,filename).group(3)
platform = re.search(pattern,filename).group(4)
encode = re.search(pattern,filename).group(5)
method = re.search(pattern,filename).group(6)
print(thread,sample,region,platform,encode,method)

meta_df = pd.read_table(tsv_path).iloc[:MAX_SAMPLE_SIZE, :].reset_index()
data = np.load(nbr_path) 
nbr_indices = data['arr_0']

with open(ref_graph_path,'rb') as f:
    reference_graph = pickle.load(f)
print("reference graph loading done")
ref_edges_num = reference_graph.number_of_edges()
reference_edges = set(
    tuple(sorted((node_1, node_2)))
    for node_1, node_2 in reference_graph.edges
)
filter_reference_edges = set(
    tuple(sorted((node_1, node_2)))
    for node_1, node_2, data in reference_graph.edges(data=True)
    if data['overlap_size'] >= 500
)

max_n_neighbors = 19
df_rows = []
read_ids = np.array(list(meta_df.index))
k_values = np.arange(1, max_n_neighbors)

neighbor_edges_nums = []
precisions = []
mean_overlap_sizes = []
filter_recall = []
recall = []
singleton = []
singleton_percentage = []
connected_component = []

for k in k_values:
    graph = OverlapGraph.from_neighbor_indices(
    neighbor_indices=nbr_indices,
    n_neighbors=k,
    read_ids=read_ids,
    require_mutual_neighbors=False)
    print(f"neighbor{k} graph construct done!")

    edges_relative_num = len(graph.edges())/nbr_indices.shape[0]
    neighbor_edges_nums.append(edges_relative_num)
    overlap_sizes = get_neighbor_overlap_bases(query_graph=graph, reference_graph=reference_graph)
    mean_overlap_sizes.append(sum(overlap_sizes)/len(overlap_sizes))
    precision = 1-(overlap_sizes.count(0)/len(overlap_sizes))
    precisions.append(precision)
    print(precision,edges_relative_num)
    query_edges = set(
        tuple(sorted((read_1, read_2))) for read_1, read_2 in graph.edges
    )
    true_positive_edges = query_edges & reference_edges
    filter_true_positive_edges = query_edges & filter_reference_edges
    recall.append(len(true_positive_edges)/len(reference_edges))
    filter_recall.append(len(filter_true_positive_edges)/len(filter_reference_edges))

    remove_false_edges(graph, reference_graph)
    singleton.append(len([node for node in graph if len(graph[node]) <= 1]))
    singleton_percentage.append(len([node for node in graph if len(graph[node]) <= 1])/reference_graph.number_of_nodes()) 
    connected_component.append(nx.number_connected_components(graph))
    
    del graph
    gc.collect()  



di = {
    'thread':thread,
    'sample':sample,
    'region':region,
    'platform':platform,
    'encode':encode,
    'method':method,
    'n_neighbors':range(1,max_n_neighbors),
    'edges_num':neighbor_edges_nums,
    'precision':precisions,
    'overlap_size':mean_overlap_sizes,
    'recall':recall,
    'filter_recall':filter_recall,
    'singleton':singleton,
    'singleton_percentage':singleton_percentage,
    'connected_component':connected_component
    }
df = pd.DataFrame(di)
df.to_csv(df_file,sep='\t',index=False)
