import sys,pickle
import pandas as pd
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/scripts")
from graph import OverlapGraph

def get_sibling_id(x: int) -> int:
    if x % 2 == 0:
        return x + 1
    else:
        return x - 1
    
def calculate_precision(overlap_candicates, G):
    TP, FP = 0, 0
    tested_edges = set()  # 用于记录已测试的边（无向图使用有序对）
    for (i,j) in overlap_candicates:
        edge = (min(i,j), max(i,j))
        edge_sib = (get_sibling_id(min(i,j)), get_sibling_id(max(i,j)))
        if edge not in tested_edges:
            tested_edges.add(edge)
            if G.has_edge(*edge) or G.has_edge(*edge_sib):
                TP += 1
            else:
                FP += 1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision

platform = sys.argv[2]
ref_graph_path = f"/home/miaocj/docker_dir/kNN-overlap-finder/data/regional_reads/CHM13/all/filter3_real_{platform}/reference_graph.gpickle"
with open(ref_graph_path,'rb') as f:
    reference_graph = pickle.load(f)
print("reference graph loading done")

meta_df = pd.read_table(f"/home/miaocj/docker_dir/kNN-overlap-finder/data/regional_reads/CHM13/all/filter3_real_{platform}/metadata.tsv.gz").reset_index()
single_orientation_df = meta_df[meta_df['read_orientation'] == '+']
name_id_dict = dict(zip(single_orientation_df['read_name'], single_orientation_df['read_id']))

overlap_candidates = []
with open(sys.argv[1], 'r') as f:
    for lines in f:
        line = lines.strip().split('\t')
        r1_idx = name_id_dict[line[0]]
        if line[2] == '1':
            r2_idx = name_id_dict[line[1]]
        else:
            r2_idx = name_id_dict[line[1]]+1
        overlap_candidates.append((r1_idx, r2_idx))

precision = calculate_precision(set(overlap_candidates), reference_graph)
print(f"Precision: {precision}")
