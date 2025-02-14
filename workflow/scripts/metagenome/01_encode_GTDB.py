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

# Build gtdb_taxonomy
def extract_groups(taxonomy):
    match = re.search(pattern, taxonomy)
    if match:
        return match.group(1)
    else:
        return None

pattern = r'^.+s__(.+ .+)$'
gtdb_taxonomy = pd.read_csv('/home/miaocj/docker_dir/data/GTDB_download/release220/taxonomy/gtdb_taxonomy.tsv',sep='\t',header=None)
gtdb_taxonomy.columns=['tax_id','taxonomy']
gtdb_taxonomy[['species']] = gtdb_taxonomy['taxonomy'].apply(lambda x: pd.Series(extract_groups(x))) ## eg: ['GCA454151','Arabidopsis thanlian']
print('gtdb_taxonomy building finished')

with open("/home/miaocj/docker_dir/data/GTDB_download/release220/skani/gtdb_all_fa/readid_gcaid.json", "rt") as json_file:
    id_dict = json.load(json_file) 
print('id_dict load finished')

fasta_file = "/home/miaocj/docker_dir/kNN-overlap-finder/data/metagenome_reference/GTDB/mer_sampled.fa"
kmer_dict = collections.defaultdict()
for i,record in enumerate(parse_fasta(fasta_file)):
    kmer_dict[record[1]] = i

print('kmer_dict building finished')

db_part_path = sys.argv[1]
batch_id = sys.argv[2] 
batch_size = 200000
feature_matrix, read_features, metadata = encode_reads(db_part_path,kmer_dict,id_dict,gtdb_taxonomy,16,int(batch_id),batch_size)
print('feature_matrix building finished')

output_npz_file = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/feature_matrix_{batch_id}.npz'
output_json_file = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/read_features_{batch_id}.npz'
output_tsv_file = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/metadata_{batch_id}.tsv.gz'

sp.save_npz(output_npz_file, feature_matrix)
with open_gzipped(output_json_file, "wt") as f:
    json.dump(read_features, f)
metadata.to_csv(output_tsv_file, index=False, sep="\t")

