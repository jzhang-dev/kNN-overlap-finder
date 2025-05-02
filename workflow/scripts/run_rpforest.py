from rpforest import RPForest
from scipy import sparse as sp
from scipy.sparse import csr_matrix
import numpy as np
from sklearn import random_projection
from sklearn.feature_extraction.text import TfidfTransformer
import sys,time,json
print(f"Config: {sys.argv[1]}, Leaf Size: {sys.argv[2]}, No Trees: {sys.argv[3]}")

def rpf(region,dim,config,leaf_size,no_trees):
    elapsed_time = {}
    fm_path = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/{region}/kmer_k11/feature_matrix.npz'
    fm = sp.load_npz(fm_path)
    start_time = time.time()
    fm = csr_matrix((np.ones_like(fm.data), fm.indices, fm.indptr), shape=fm.shape)
    fm = TfidfTransformer(use_idf=True, smooth_idf=True).fit_transform(fm)
    elapsed_time['tfidf'] = time.time() - start_time
    print('IDF done!')

    start_time = time.time()
    reducer = random_projection.SparseRandomProjection(n_components=dim)
    _embedding = reducer.fit_transform(fm)
    embedding = _embedding.toarray()
    elapsed_time['dimension_reduction'] = time.time() - start_time
    print('dimensional reduction done!')

    start_time = time.time()
    all_neighbors = []
    model = RPForest(leaf_size=int(leaf_size), no_trees=int(no_trees))
    model.fit(embedding)
    for i in range(0,embedding.shape[0]):
        nns = model.query(embedding[i,:], 20)
        all_neighbors.append(nns)
    nbr = np.array(all_neighbors)
    elapsed_time['nearest_neighbors'] = time.time() - start_time
    print(nbr)

    nbr_path = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation64/{region}/kmer_k11/RPForest_config{str(config)}/RPForest_Cosine_SparseRP_{str(dim)}d_IDF_nbr_matrix.npz'
    np.savez(nbr_path, nbr)

    time_path  =f'/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation64/{region}/kmer_k11/RPForest_config{str(config)}/RPForest_Cosine_SparseRP_{str(dim)}d_IDF_nn_time.json'
    with open(time_path, 'w', encoding='utf-8') as f:
        json.dump(elapsed_time, f, ensure_ascii=False)
# for region in ['CHM13/IGK/real_cyclone','CHM13/HLA/real_cyclone','CHM13/IGK/real_ONT','CHM13/HLA/real_ONT','yeast/chr4_1M/real_cyclone']:
#     print(region)
#     rpf(region,1500)
rpf(sys.argv[4],3000,sys.argv[1],sys.argv[2],sys.argv[3])
# for region in ['CHM13/chr1_248M/real_cyclone','CHM13/chr1_248M/real_ONT']:
#     print(region)
#     rpf(region,3000)