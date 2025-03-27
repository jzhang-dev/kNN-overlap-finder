from evaluate import NearestNeighborsConfig
from dim_reduction import SpectralEmbedding, scBiMapEmbedding,GaussianRandomProjection,SparseRandomProjection,SimHash_Dimredu,PCA,isomap,umapEmbedding,Split_GRP
from nearest_neighbors import ExactNearestNeighbors,PAFNearestNeighbors,SimHash,HNSW,ProductQuantization,NNDescent,WeightedLowHash,IVFProductQuantization
max_bucket_size = 20
def parse_string_to_config(input_string: str,nearest_neighbors_parameter: dict) -> NearestNeighborsConfig:
    # 将字符串按下划线分割
    mydict = {  
    'Exact':ExactNearestNeighbors,
    'SimHash':SimHash,
    'HNSW':HNSW,
    'PQ':ProductQuantization,
    'NNDescent':NNDescent,
    'MinHash':WeightedLowHash,
    'IVFPQ':IVFProductQuantization,
    'Euclidean':'euclidean',
    'Cosine':'cosine',
    'SplitGRP':Split_GRP,
    'GaussianRP':GaussianRandomProjection,
    'SparseRP':SparseRandomProjection,
    'scBiMap':scBiMapEmbedding,
    'Spectural':SpectralEmbedding,
    'SimHash_Dimredu':SimHash_Dimredu,
    'PCA':PCA,
    'isomap':isomap,
    'umap':umapEmbedding,
    }
    parts = input_string.split('_')
    if len(parts) == 5:
        nearest_neighbors_method = mydict[parts[0]]
        description = input_string
        tfidf = parts[4]
        dimension_reduction_method = mydict[parts[2]]
        dimension_reduction_kw = dict(n_dimensions=int(parts[3][:-1]))  # 去掉最后的'd'
        nearest_neighbors_kw = dict(metric=mydict[parts[1]])
        nearest_neighbors_kw.update(nearest_neighbors_parameter)
        myNearestNeighborsConfig = NearestNeighborsConfig(
        nearest_neighbors_method=nearest_neighbors_method,
        description=description,
        tfidf=tfidf,
        dimension_reduction_method=dimension_reduction_method,
        dimension_reduction_kw=dimension_reduction_kw,
        nearest_neighbors_kw=nearest_neighbors_kw
    )
    elif len(parts) == 4:
        nearest_neighbors_method = mydict[parts[0]]
        tfidf = parts[3]
        if parts[0] == 'MinHash':
            myNearestNeighborsConfig = NearestNeighborsConfig(
            nearest_neighbors_method=WeightedLowHash,
            description=input_string,
            tfidf=tfidf,
            nearest_neighbors_kw=dict(
            lowhash_count=20,
            max_bucket_size=max_bucket_size,
            repeats=100,
            seed=458,
            ))
        elif parts[0] == 'SimHash':
            myNearestNeighborsConfig=NearestNeighborsConfig(
            nearest_neighbors_method=SimHash,
            description=input_string,
            tfidf=tfidf)
        else:       
            nearest_neighbors_kw = dict(metric=mydict[parts[1]])
            myNearestNeighborsConfig = NearestNeighborsConfig(
            nearest_neighbors_method=nearest_neighbors_method,
            description=input_string,
            tfidf=tfidf,
            nearest_neighbors_kw=nearest_neighbors_kw)
    # 创建并返回 NearestNeighborsConfig 实例
    return myNearestNeighborsConfig

