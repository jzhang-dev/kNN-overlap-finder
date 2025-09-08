from pathlib import Path
from Bio import SeqIO
import collections
import gzip
import re
import json

dir_path = Path("/home/miaocj/docker_dir/data/GTDB_download/release220/skani/database/")
pattern=r'GC\D_\d+\.\d'
id_dict = collections.defaultdict() ##{read_id1,GCA0005454}
for file in dir_path.rglob("*"):
    if file.is_file():
        ref_id=re.search(pattern,str(file))[0]
        if ref_id[:3] == 'GCA':
            ref_id = 'GB_'+ref_id
        else:
            ref_id = 'RS_'+ref_id
        with gzip.open(file, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                id_dict[record.id] = ref_id

pattern = r'^.+s__(.+ .+)$'
gtdb_taxonomy = pd.read_csv('/home/miaocj/docker_dir/data/GTDB_download/release220/taxonomy/gtdb_taxonomy.tsv',sep='\t',header=None)
gtdb_taxonomy.columns=['tax_id','taxonomy']
gtdb_taxonomy[['spe']] = gtdb_taxonomy['taxonomy'].apply(lambda x: pd.Series(extract_groups(x)))

with open("/home/miaocj/docker_dir/data/GTDB_download/release220/skani/gtdb_all_fa/readid_gcaid.json", "w") as json_file:
    json.dump(id_dict, json_file) 

# ind_ref = collections.defaultdict() ##{index,GCA0005454}
# with gzip.open('/home/miaocj/docker_dir/data/GTDB_download/release220/skani/gtdb_all_fa/GTDB_database.fna.gz', "rt") as handle:
#     for index,record in enumerate(SeqIO.parse(handle, "fasta")):
#         ind_ref[index] = id_dict[record.id]

# with open("/home/miaocj/docker_dir/data/GTDB_download/release220/skani/gtdb_all_fa/index_refid.json", "w") as json_file:
#     json.dump(ind_ref, json_file) 