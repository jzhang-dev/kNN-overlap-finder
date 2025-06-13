import pickle, sys
import numpy as np
import collections
import pandas as pd
import os, re
import json

sys.path.append('/home/miaocj/docker_dir/kNN-overlap-finder/scripts')
def get_sibling_id(x: int) -> int:
    if x % 2 == 0:
        return x + 1
    else:
        return x - 1
    
class FEDRANNNearestNeighbors():
    def get_neighbors(
        self,
        overlap_txt: str,
        n_neighbors: int, 
        ) -> np.ndarray:
        # Calculate cumulative alignment lengths
        distance_dict = collections.defaultdict(collections.Counter)
        i=0
        with open(overlap_txt,'rt') as f:
            next(f)
            for line in f:
                li = line.strip().split('\t')
                if len(li) == 4:
                    i1 = int(li[0])
                    i2 = int(li[1])
                    sibling_i1 = get_sibling_id(i1)
                    sibling_i2 = get_sibling_id(i2)
                    distance_dict[i1][i2] = int(li[3])
                    distance_dict[i2][i1] = int(li[3])
                    distance_dict[sibling_i1][sibling_i2] = int(li[3])
                    distance_dict[sibling_i2][sibling_i1] = int(li[3])
        max_read_id = max(distance_dict.keys())  
        nbr_matrix = np.empty((max_read_id+1, n_neighbors), dtype=np.int64)
        nbr_matrix[:, :] = -1
        for i in range(max_read_id+1):
            row_nbr_dict = {
                j: distance
                for j, distance in distance_dict[i].items()
            }
            neighbors = list( 
                sorted(row_nbr_dict, key=lambda x: row_nbr_dict[x], reverse=False)
            )[:n_neighbors]
            nbr_matrix[i, : len(neighbors)] = neighbors
        return nbr_matrix
    
def calculate_precision(neighbor_matrix, G):
    precisions = []
    for n in [6,12,18]:
        TP, FP = 0, 0
        tested_edges = set()  # 用于记录已测试的边（无向图使用有序对）
        for i, neighbors in enumerate(neighbor_matrix):
            for j in neighbors[:n]:
                edge = (min(i,j), max(i,j))
                if edge not in tested_edges:
                    tested_edges.add(edge)
                    if G.has_edge(i, j):
                        TP += 1
                    else:
                        FP += 1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        precisions.append(precision)
        print(f"Precision with {n} neighbors: {precision}")
    return precisions

def statistic_tp_numbers_of_reads(neighbor_matrix, G):
    one_read_tp_neighbors_numbers = []
    for i, neighbors in enumerate(neighbor_matrix):
        one_read_tp_neighbors = 0
        for j in neighbors[:20]:
            if G.has_edge(i, j):
                one_read_tp_neighbors += 1
        one_read_tp_neighbors_numbers.append(one_read_tp_neighbors)
    one_read_tp_neighbors_counter = collections.Counter(one_read_tp_neighbors_numbers)
    return one_read_tp_neighbors_counter

def main():

    suffix = sys.argv[1]
    platform = sys.argv[2]
    species = sys.argv[3]

    overlap_txt_path = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/FEDRANN/{species}/all/filter3_real_{platform}/hash_k21/alignment_{suffix}.txt'
    new_metadata_path = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/FEDRANN/{species}/all/filter3_real_{platform}/hash_k21/metadata_{suffix}.tsv'
    nbr_path = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/FEDRANN/{species}/all/filter3_real_{platform}/nbr_matrix_{suffix}.npz'
    stat_path = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/FEDRANN/{species}/all/filter3_real_{platform}/part_stat_{suffix}.tsv'
    ref_graph_path = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/regional_reads/{species}/all/filter3_real_{platform}/reference_graph.gpickle'
    metadata_old_path = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/regional_reads/{species}/all/filter3_real_{platform}/metadata.tsv.gz'

    if os.path.exists(overlap_txt_path) and os.path.exists(new_metadata_path) and os.path.exists(ref_graph_path) and os.path.exists(metadata_old_path):
        print("All input files exist, proceeding with evaluation.")
    else:
        print("One or more input files are missing. Please check the paths.")
        print(overlap_txt_path, new_metadata_path, ref_graph_path, metadata_old_path)
        return
    
    if os.path.exists(nbr_path):
        print(f"Neighbor matrix already exists at {nbr_path}. Loading from file.")
        nbr_matrix = np.load(nbr_path)['arr_0']
    else:
        nbr_matrix = FEDRANNNearestNeighbors().get_neighbors(
            overlap_txt=overlap_txt_path,
            n_neighbors=20
        )
        print(nbr_matrix)
        print("neighbor matrix loading done")
        # nbr_matrix = np.load(nbr_path)['arr_0']

        metadata_mcj = pd.read_csv(metadata_old_path, sep='\t')
        metadata_zjy = pd.read_csv(new_metadata_path, sep='\t')
        metadata_mcj['strand'] = metadata_mcj['read_orientation'].replace({'+': 0, '-': 1})
        merged_df = pd.merge(
            metadata_mcj, 
            metadata_zjy, 
            on=['read_name', 'strand'], 
            how='inner'
        )
        id_transfor_dict = dict(zip(merged_df['index'],merged_df['read_id']))
        new_nbr_matrix = np.empty(nbr_matrix.shape, dtype=np.int32)
        new_nbr_matrix[:, :] = -1
        for i in range(nbr_matrix.shape[0]):
            new_nbr_matrix[id_transfor_dict[i], :len(nbr_matrix[i])] =  [id_transfor_dict[x] for x in nbr_matrix[i]]
        np.savez(nbr_path, nbr_matrix)

    with open(ref_graph_path,'rb') as f:
        reference_graph = pickle.load(f)
    print("reference graph loading done")

    precisions = calculate_precision(new_nbr_matrix, reference_graph)
    one_read_tp_neighbors_counter = statistic_tp_numbers_of_reads(new_nbr_matrix, reference_graph)
    print("One read TP neighbors counter:", one_read_tp_neighbors_counter)
    with open(stat_path, 'w') as f:
        json.dump(
            {
                'precisions': precisions,
                'one_read_tp_neighbors_counter': dict(one_read_tp_neighbors_counter)
            },
            f, indent=4
        )

if __name__ == "__main__":
    main()