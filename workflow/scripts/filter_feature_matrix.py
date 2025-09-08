import scipy as sp
import numpy as np
import pandas as pd
import json,sys,gzip
fm_file = sys.argv[1]
read_name_file = sys.argv[2]
meta_file =sys.argv[3]
out_fm_file = sys.argv[4]

fm = sp.sparse.load_npz(fm_file)
meta_df = pd.read_table(meta_file).reset_index()

read_to_indices = meta_df.groupby('read_name').indices
filter_row_index = []
with open(read_name_file,'rt') as f2:
    for lines in f2:
        line = lines.strip()
        indices = read_to_indices.get(line, [])
        filter_row_index.extend(indices)
filter_fm = fm[filter_row_index,:]

sp.save_npz(out_fm_file, filter_fm)