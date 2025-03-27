import scipy.sparse as sp
import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import sys

fm_list = []
meta_list = []

dir_path = "/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/all_rp/"
for i in range(1,88):
    fm_filename = 'feature_matrix_' + str(i) + '.npz'
    # meta_filename = 'metadata_' + str(i) + '.tsv.gz'
    fm_part = sp.load_npz(dir_path+fm_filename)
    print(fm_filename)
    fm_list.append(fm_part)
    # meta_part = pd.read_csv(dir_path+meta_filename,sep='\t')
    # print(meta_filename)
    # meta_list.append(meta_part)

fm = sp.vstack(fm_list)
print('vstack DONE!')
fm = fm.astype(np.uint16)
output_ref_npz_file = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/all_rp/feature_matrix_rp_1.npz'
sp.save_npz(output_ref_npz_file, fm)

# meta = pd.concat(meta_list,axis=0)
# meta = meta.reset_index()
# output_meta_file = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/metadata_all.tsv'
# meta.to_csv(output_meta_file,index=False,sep='\t')