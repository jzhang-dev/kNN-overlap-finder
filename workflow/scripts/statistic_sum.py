#!python
import os  
import sys
import pickle
import itertools
import statistics
import re 
import pandas as pd

# 输入文件夹路径  
sample = sys.argv[1]
file_suffix = sys.argv[2]
all_folder_path = ['CHM13/IGK/real_cyclone','CHM13/HLA/real_cyclone','CHM13/chr1_248M/real_cyclone',
            'CHM13/IGK/real_ONT','CHM13/HLA/real_ONT','CHM13/chr1_248M/real_ONT',
            'ara/chr1_30M/real_ONT','rice/chr1_43M/real_ONT_new','yeast/chr4_1M/real_cyclone']
if sample == 'all':
    folder_path = all_folder_path
elif sample in  all_folder_path:
    folder_path = [sample]
else:
    sys.exit('wrong sample')
    
# 输出文件名  
output_file1 = f"summarize_stat_{file_suffix}.tsv"

all_di = {}
all_integral_precision = []
all_integral_mean = []
all_methods = []
all_samples = []
all_neighbors = []
all_configs = []

def process_overlap_sizes_file(filename: str,config: str ='default'):
    pattern = r'/[^/]+_o'
    integral_precision,integral_mean,samples,methods,neighbors,configs = [],[],[],[],[],[]
    with open(filename,'rb') as f:
        neighbor_overlap_sizes = pickle.load(f)
        method = re.search(pattern,filename)[0][1:-2]
        print(method)
        if 'HNSW' in method:
            for i in range(len(neighbor_overlap_sizes[:-1])):
                nested_list = neighbor_overlap_sizes[:-1][:i+1]
                flatten_list = list(itertools.chain(*nested_list))
                integral_precision.append((len(flatten_list)-flatten_list.count(0))/len(flatten_list))
                integral_mean.append(statistics.mean(flatten_list))
                samples.append(folder)
                methods.append(method)
                neighbors.append(i+1)
                configs.append(config)
    return integral_precision,integral_mean,samples,methods,neighbors,configs

# 遍历文件夹  
for folder in folder_path:
    dir_path = os.path.join(folder,'kmer_k16')
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            if os.path.isdir(os.path.join(folder,'kmer_k16',filename)):
                for config_filename in os.listdir(os.path.join(folder,'kmer_k16',filename)):
                    if config_filename.endswith("overlap_sizes.pkl"):  
                        print(os.path.join(folder,'kmer_k16',filename))
                        integral_precision,integral_mean,samples,methods,neighbors,configs = process_overlap_sizes_file(
                            os.path.join(folder,'kmer_k16',filename,config_filename),filename)
                        all_integral_precision += integral_precision
                        all_integral_mean += integral_mean
                        all_methods += methods
                        all_samples += samples
                        all_neighbors += neighbors
                        all_configs += configs
            elif filename.endswith("overlap_sizes.pkl"): 
                integral_precision,integral_mean,samples,methods,neighbors,configs = process_overlap_sizes_file(
                    os.path.join(folder,'kmer_k16',filename))
                all_integral_precision += integral_precision
                all_integral_mean += integral_mean
                all_methods += methods
                all_samples += samples
                all_neighbors += neighbors
                all_configs += configs
di = {'sample':all_samples,'method':all_methods,'config':all_configs,'n_neighbors':all_neighbors,
      'integral_precision':all_integral_precision,'integral_mean':all_integral_mean}
df = pd.DataFrame(di)
df.to_csv(output_file1,sep='\t')
