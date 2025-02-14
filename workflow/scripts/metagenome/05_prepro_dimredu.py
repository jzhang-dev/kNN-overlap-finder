import sys
sys.path.append('/home/miaocj/docker_dir/SeqNeighbor')  
sys.path.append('../../../scripts')
from meta_nearest_neighbors import ExactNearestNeighbors,HNSW,NearestNeighborsConfig,compute_nearest_neighbors
from meta_str2config import parse_string_to_config
import numpy as np
import scipy.sparse as sp

ref_fm = sp.load_npz('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/feature_matrix_all.npz')
que_fm = sp.load_npz('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/sample1.npz')
nbr_path = '/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/metagenome/GTDB/sample1/nbr_matrix.npz'
print(f'GTDB fm size: {ref_fm.shape}')
print(f'sample1 fm size: {que_fm.shape}')
method = sys.argv[1]
config = parse_string_to_config(method)
nbr_path = '/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/metagenome/GTDB/sample1/'+method+'_nbr_matrix.npz'
neighbor_indices, elapsed_time, peak_memory = compute_nearest_neighbors(
ref=ref_fm,
que=que_fm,
config=config,
n_neighbors=3)
np.savez(nbr_path, neighbor_indices)

# tar_train,que_fit = tfidf_qvt(ref_fm,que_fm,'IDF')
# print('preprocess finished')
# query_fm_dim = GaussianRP().transform(que_fit, n_dimensions=500)
# print('query dimension reduction finished')
# taget_fm_dim = GaussianRP().transform(tar_train, n_dimensions=500)
# print('database dimension reduction finished')
# neighbor_indices  = HNSW().get_neighbors(taget_fm_dim,query_fm_dim, n_neighbors=2)
# print('get neighbors finished')
# print(neighbor_indices)
