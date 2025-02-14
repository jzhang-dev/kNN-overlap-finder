import scipy.sparse as sp
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import sys

fm_list = []
pattern = r"^feature_matrix_\d+\.npz$"
dir_path = "/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/"
for filename in os.listdir(dir_path):
    if re.match(pattern, filename):
        print(filename)
        fm_part = sp.load_npz(dir_path+filename)
        fm_list.append(fm_part)
fm = sp.vstack(fm_list)
output_ref_npz_file = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/feature_matrix_all.npz'
sp.save_npz(output_ref_npz_file, fm)

