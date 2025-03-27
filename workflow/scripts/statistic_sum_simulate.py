#!python
import os  
import sys
import pickle
import itertools
import statistics
import re 
import pandas as pd

# 输入文件夹路径  
# sample = sys.argv[1]
# all_folder_path = ['CHM13/IGK','CHM13/HLA','rice/chr1_43M']
# if sample == 'all':
#     folder_path = all_folder_path
# elif sample in  all_folder_path:
#     folder_path = [sample]
# else:
#     sys.exit('wrong sample')
    
# 输出文件名  
output_file1 = f"summarize_stat_pbsim.tsv"

pattern = r'.+_o'
all_di = {}
integral_precision = []
integral_mean = []
methods = []
samples = []
neighbors = []
feature = []
# 遍历文件夹  
for folder in ['rice/chr1_43M','CHM13/IGK']:
    for dir_path in os.listdir(folder):
        if dir_path.startswith('pbsim') and dir_path.endswith('dep'):
            print(dir_path)
            dir_path = os.path.join(folder,dir_path,'kmer_k16')
            for filename in os.listdir(dir_path):
                if filename.endswith("overlap_sizes.pkl"): 
                    with open(dir_path+'/'+filename,'rb') as f:
                        #print(dir_path+'/'+filename)
                        neighbor_overlap_sizes = pickle.load(f)
                        method = re.search(pattern,filename)[0][:-2]
                        for i in range(len(neighbor_overlap_sizes[:-1])):
                            nested_list = neighbor_overlap_sizes[:-1][:i+1]
                            flatten_list = list(itertools.chain(*nested_list))
                            integral_precision.append((len(flatten_list)-flatten_list.count(0))/len(flatten_list))
                            integral_mean.append(statistics.mean(flatten_list))
                            samples.append(folder)
                            feature.append(dir_path)
                            methods.append(method)
                            neighbors.append(i+1)
di = {'sample':samples,'feature':feature,'method':methods,'n_neighbors':neighbors,'integral_precision':integral_precision,'integral_mean':integral_mean}
df = pd.DataFrame(di)
df.to_csv(output_file1,sep='\t')
