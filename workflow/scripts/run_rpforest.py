from rpforest import RPForest
from scipy import sparse as sp
import numpy as np
from sklearn import random_projection
from sklearn.feature_extraction.text import TfidfTransformer
import sys
print(f"Config: {sys.argv[1]}, Leaf Size: {sys.argv[2]}, No Trees: {sys.argv[3]}")

def rpf(region,dim,config,leaf_size,no_trees):
    fm_path = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/{region}/kmer_k16/feature_matrix.npz'
    fm = sp.load_npz(fm_path)
    fm[fm > 0] = 1
    fm = TfidfTransformer(use_idf=True, smooth_idf=True).fit_transform(fm)
    print('IDF done!')
    reducer = random_projection.SparseRandomProjection(n_components=dim)
    _embedding = reducer.fit_transform(fm)
    embedding = _embedding.toarray()
    print('dimensional reduction done!')
    all_neighbors = []
    model = RPForest(leaf_size=int(leaf_size), no_trees=int(no_trees))
    model.fit(embedding)
    for i in range(0,embedding.shape[0]):
        nns = model.query(embedding[i,:], 20)
        all_neighbors.append(nns)
    nbr = np.array(all_neighbors)
    print(nbr)
    nbr_path = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/{region}/kmer_k16/RPForest_config{str(config)}/RPForest_SparseRP_{str(dim)}d_nbr_matrix.npz'
    np.savez(nbr_path, nbr)

# for region in ['CHM13/IGK/real_cyclone','CHM13/HLA/real_cyclone','CHM13/IGK/real_ONT','CHM13/HLA/real_ONT','yeast/chr4_1M/real_cyclone']:
#     print(region)
#     rpf(region,1500)
rpf('CHM13/chr1_248M/real_cyclone',3000,sys.argv[1],sys.argv[2],sys.argv[3])
# for region in ['CHM13/chr1_248M/real_cyclone','CHM13/chr1_248M/real_ONT']:
#     print(region)
#     rpf(region,3000)