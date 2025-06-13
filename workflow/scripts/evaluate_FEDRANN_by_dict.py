import pickle, sys
import numpy as np
import collections
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

def statistic_fp_numbers_of_reads(neighbor_matrix, G):
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
    overlap_txt_path = sys.argv[1]
    # ref_graph_path = sys.argv[2]
    nbr_path = sys.argv[2]

    nbr_matrix = FEDRANNNearestNeighbors().get_neighbors(
        overlap_txt=overlap_txt_path,
        n_neighbors=20
    )
    print(nbr_matrix)
    print("neighbor matrix loading done")
    np.savez(nbr_path, nbr_matrix)
    # nbr_matrix = np.load(nbr_path)['arr_0']

    # with open(ref_graph_path,'rb') as f:
    #     reference_graph = pickle.load(f)
    # print("reference graph loading done")

    # precisions = calculate_precision(nbr_matrix, reference_graph)
    # one_read_tp_neighbors_counter = statistic_fp_numbers_of_reads(nbr_matrix, reference_graph)
    # print("One read TP neighbors counter:", one_read_tp_neighbors_counter)

if __name__ == "__main__":
    main()