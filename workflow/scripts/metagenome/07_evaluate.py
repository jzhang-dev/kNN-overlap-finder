from pathlib import Path
from Bio import SeqIO
import collections
import gzip
import re
import json
import pandas as pd
import numpy as np

def extract_groups(taxonomy):
    match = re.search(pattern, taxonomy)
    if match:
        return match.group(1)
    else:
        return None

# ------------------ 处理数据库------------------
pattern = r'^.+s__(.+ .+)$'
gtdb_taxonomy = pd.read_csv('/home/miaocj/docker_dir/data/GTDB_download/release220/taxonomy/gtdb_taxonomy.tsv',sep='\t',header=None)
gtdb_taxonomy.columns=['tax_id','taxonomy']
gtdb_taxonomy[['spe']] = gtdb_taxonomy['taxonomy'].apply(lambda x: pd.Series(extract_groups(x))) ## eg: ['GCA454151','Arabidopsis thanlian']

with open("/home/miaocj/docker_dir/data/GTDB_download/release220/skani/gtdb_all_fa/index_refid.json", "r") as json_file:
    ind_ref = json.load(json_file) ##eg {index,'GCA0005454'}
print(len(ind_ref))
print('finished database process')

## -------------处理模拟样本-----------
id_readname = {}
file = "/home/miaocj/docker_dir/kNN-overlap-finder/data/metagenome_reference/GTDB/mer_sampled.fa"
with open(file, "rt") as handle:
    for i,record in enumerate(SeqIO.parse(handle, "fasta")):
        id_readname[i]=record.id ## {index:read_id} 获取index和id之间的关系

print('finished sample0 id dict generation')

## ------------获取每个read_id 预测所属的物种--------
method = 'HNSW_Cosine_GaussianRP_500d_IDF'
nbr_indice = np.load('/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/metagenome/GTDB/sample1/'+method+'_nbr_matrix.npz')['arr_0']
pre_taxs = []
read_names = []
for i, row in enumerate(nbr_indice[:,:1]):
    if i % 10000 == 0:
        print(i)
    neighbor_indice = row[0]
    gca_id = ind_ref[neighbor_indice]
    tax =  gtdb_taxonomy.loc[gtdb_taxonomy['tax_id'] == gca_id, 'spe']
    pre_taxs.append(tax)
    read_names.append(id_readname[i])
predict_dict = {'read_id':read_names,'predict_species':pre_taxs}
predict_df = pd.DataFrame(predict_dict)
predict_df[['genus','species']]= predict_df['spe'].str.split(' ',expand=True,n=1)
predict_df.to_csv('/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/metagenome/GTDB/sample1/'+method+'_predict_taxonomy.npz')
    
##保存read name 应该放在编码时

#read_tax = pd.read_csv('/home/miaocj/docker_dir/data/cami_download/2018.01.23_11.53.11_sample_0/reads/read_tax.tsv',sep='\t') 
##真实的
# sample_tax = pd.read_csv('/home/miaocj/docker_dir/data/cami_download/2018.01.23_11.53.11_sample_0/reads/abundance_lineage.tsv',sep='\t')
# read_mapping = pd.read_csv('/home/miaocj/docker_dir/data/cami_download/2018.01.23_11.53.11_sample_0/reads/reads_mapping.tsv',sep='\t')
# read_mapping.columns = ['read_id','genome_ID','tax_id','other']
# read_tax = pd.merge(read_mapping,sample_tax,how='left',on='genome_ID').iloc[:,[0,8]]
# read_tax.columns=['read_id','spe']
# read_tax['spe'] = read_tax['spe'].fillna(' ')
# read_tax[['genus','species']]= read_tax['spe'].str.split(' ',expand=True,n=1)
# read_tax = read_tax.iloc[:,[0,2,3]] 
# read_tax.to_csv('/home/miaocj/docker_dir/data/cami_download/2018.01.23_11.53.11_sample_0/reads/read_tax.tsv',sep='\t',index=False)