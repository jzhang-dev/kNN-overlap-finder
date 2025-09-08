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
        for record in parse_fasta(str(file)):
            id_dict[record[0]] = ref_id
with open("/home/miaocj/docker_dir/data/GTDB_download/release220/skani/gtdb_all_fa/readid_gcaid.json", "wt") as json_file:
    json.dump(id_dict, json_file) 