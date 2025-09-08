import sys
sys.path.append('/home/miaocj/docker_dir/SeqNeighbor')  
sys.path.append('../../../scripts')
sys.path.append('/home/miaocj/docker_dir/kNN-overlap-finder/scripts')
import numpy as np
import scipy.sparse as sp
import pickle
from sklearn import random_projection
from meta_nearest_neighbors import HNSW
from sklearn.feature_extraction.text import TfidfTransformer
import time

# que_fm = sp.load_npz('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/sample1_feature_matrix.npz')
# reducer = random_projection.SparseRandomProjection(n_components=500)
# _que = reducer.fit_transform(que_fm)
# try:
#     _que = _que.astype(np.float16)
# except Exception as e:
#     pickle.dump(_que,open('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/sample1/sample1_3000d.pkl','wb'),protocol=4)
# else:
#     pickle.dump(_que,open('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/sample1/sample1_3000d.pkl','wb'),protocol=4)

elapsed_time = {}
ref_fm = sp.load_npz('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/all/feature_matrix_all_unit16.npz')
ref_fm_rp = sp.load_npz('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/all_rp/feature_matrix_rp_1.npz')
print('loading done')
_data = sp.vstack([ref_fm,ref_fm_rp])
print('vstack done')
print(_data.shape)
start_time = time.time()
binary_matrix = sp.csr_matrix((np.ones_like(_data.data), _data.indices, _data.indptr), shape=_data.shape)
col_sums = binary_matrix.sum(axis=0).A1
N = binary_matrix.shape[0]
idf = np.log((N + 1) / (col_sums + 1)) + 1 
idf = idf.astype(binary_matrix.dtype)
_data = binary_matrix.multiply(idf) 
elapsed_time['tfidf'] = time.time() - start_time
print('idf done')
start_time = time.time()
reducer = random_projection.SparseRandomProjection(n_components=3000) 
_ref = reducer.fit_transform(_data)
elapsed_time['dimension_reduction'] = time.time() - start_time
print('dimension reduction done')

que = sp.load_npz('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/kmer_k11/sample1_feature_matrix.npz')






time_path = '/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation64/metagenome/GTDB/sample1/nn_time.json'
with open(time_path, 'w', encoding='utf-8') as f:
    json.dump(elapsed_time, f, ensure_ascii=False)

try:
    _ref = _ref.astype(np.float16)
except Exception as e:
    pickle.dump(_ref,open('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/sample1/GTDB_IDF_3000d.pkl','wb'),protocol=4)
else:
    pickle.dump(_ref,open('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/sample1/GTDB_IDF_3000d.pkl','wb'),protocol=4)


## Finding Nearest neighbor
# que = pickle.load(open('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/dim2000/sample1_2000d.pkl','rb'))
# ref = pickle.load(open('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/dim2000/GTDB_2000d.pkl','rb'))
# neighbor_indices = HNSW().get_neighbors(ref,que,n_neighbors=2,metric='cosine',M=512)
# nbr_path = '/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/metagenome/GTDB/HSNW_Cosine_SRP2000_None_nbr_matrix.npz'
# np.savez(nbr_path, neighbor_indices)

# print(f'GTDB fm size: {ref_fm.shape}')
# print(f'sample1 fm size: {que_fm.shape}')
# method = 'HNSW_Cosine_SparseRP_3000d_IDF'
# config = parse_string_to_config(method)
# _ref, _que, elapsed_time, peak_memory = do_proprecess_dim(
#     ref=ref_fm,
#     que=que_fm,
#     config=config,
# )
# print(f'reference matrix max: {_ref.max()}, min {_ref.min()}')
# print(f'query matrix shape: {_que.max()}, min {_que.min()}')

# _que = _que.astype(np.float16)
#np.savez_compressed('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/sample1/sample1_IDF_3000d.npz', _que)
#np.savez_compressed('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/sample1/GTDB_IDF_3000d.npz', _ref)
# neighbor_indices, elapsed_time, peak_memory = compute_nearest_neighbor(_ref, _que,elapsed_time, peak_memory, config,2)
# nbr_path = '/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/metagenome/GTDB/sample1/'+method+'_nbr_matrix.npz'
# np.savez(nbr_path, neighbor_indices)

# tar_train,que_fit = tfidf_qvt(ref_fm,que_fm,'IDF')
# print('preprocess finished')
# query_fm_dim = GaussianRP().transform(que_fit, n_dimensions=500)
# print('query dimension reduction finished')
# taget_fm_dim = GaussianRP().transform(tar_train, n_dimensions=500)
# print('database dimension reduction finished')
# neighbor_indices  = HNSW().get_neighbors(taget_fm_dim,query_fm_dim, n_neighbors=2)
# print('get neighbors finished')
# print(neighbor_indices)
