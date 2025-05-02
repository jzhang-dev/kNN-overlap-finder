from Bio import SeqIO
import collections
import numpy as np
import scipy.sparse as sp
import sys 
import re,json,gzip
from pathlib import Path
import pandas as pd

sys.path.append('/home/miaocj/docker_dir/kNN-overlap-finder/scripts')
sys.path.append('./')
from encode_function import encode_reads
from accelerate import open_gzipped,parse_fasta
import concurrent.futures
import pickle


def extract_groups(taxonomy):
    pattern = r'^.+s__(.+ .+)$'
    match = re.search(pattern, taxonomy)
    if match:
        return match.group(1)
    else:
        return None

def process_batch(batch_id, db_part_path, kmer_dict, id_dict, gtdb_taxonomy):
    # 构建特征矩阵
    feature_matrix, metadata = encode_reads(
        db_part_path, kmer_dict, id_dict, gtdb_taxonomy, 11
    )
    print(f'feature matrix shape: {feature_matrix.shape}')
    print(f'Batch {batch_id}: feature_matrix building finished')

    # 定义输出文件路径
    output_npz_file = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/all_rp/kmer_k11/feature_matrix_{batch_id}.npz'
    output_tsv_file = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/all_rp/kmer_k11/metadata_{batch_id}.tsv.gz'

    # 保存文件
    sp.save_npz(output_npz_file, feature_matrix)
    metadata.to_csv(output_tsv_file, index=False, sep="\t")
    print(f'Batch {batch_id}: files saved successfully')

def main():
    # 输入参数
    gtdb_taxonomy = pd.read_csv('/home/miaocj/docker_dir/data/GTDB_download/release220/taxonomy/gtdb_taxonomy.tsv',sep='\t',header=None)
    gtdb_taxonomy.columns=['tax_id','taxonomy']
    gtdb_taxonomy[['species']] = gtdb_taxonomy['taxonomy'].apply(lambda x: pd.Series(extract_groups(x))) ## eg: ['GCA454151','Arabidopsis thanlian']
    gtdb_taxonomy = gtdb_taxonomy.set_index('tax_id')
    print('gtdb_taxonomy building finished')

    with open("/home/miaocj/docker_dir/data/GTDB_download/release220/skani/gtdb_all_fa/readid_gcaid.json", "rt") as json_file:
        id_dict = json.load(json_file) 
    print('id_dict load finished')

    with open("/home/miaocj/docker_dir/kNN-overlap-finder/data/metagenome_reference/GTDB/kmer_dict.pkl", "rb") as file:
        kmer_dict = pickle.load(file)
    print('kmer_dict load finished')
    
    batch_id = sys.argv[1]
    db_part_path = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/metagenome_reference/GTDB/GTDB_rp/GTDB_rp_{batch_id}.fa'
    process_batch(batch_id,db_part_path,kmer_dict, id_dict, gtdb_taxonomy)

if __name__ == "__main__":
    main()