import hnswlib
import pickle
import numpy as np
from scipy import sparse
## 构建HNSW index
ref = pickle.load(open('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/sample1/GTDB_IDF_2000d.pkl','rb'))
if sparse.issparse(ref):
    ref = ref.toarray()
p = hnswlib.Index(space='cosine', dim=ref.shape[1]) 
p.set_num_threads(1)
p.init_index(max_elements=ref.shape[0], ef_construction=200, M=512)
ids = np.arange(ref.shape[0])
p.add_items(ref,ids)
p.save_index("/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/dim2000/GTDB_2000d_HNSW_index.bin")