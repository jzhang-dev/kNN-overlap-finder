#!python
import os  
import sys
import pickle
import itertools
import statistics
import re 
import pandas as pd

# 输入文件夹路径  
folder_path = ['CHM13/IGK/real_cyclone','CHM13/HLA/real_cyclone','CHM13/chr1_248M/real_cyclone',
               'CHM13/IGK/real_ONT','CHM13/HLA/real_ONT','CHM13/chr1_248M/real_ONT',
               'ara/chr1_30M/real_ONT','rice/chr1_43M/real_ONT_new','yeast/chr4_1M/real_cyclone']
# 输出文件名  
output_file = "summarize_stat_n6.csv"
output_file1 = "summarize_stat_all_n.tsv"

pattern = r'.+_o'
all_di = {}
integral_precision = []
integral_mean = []
methods = []
samples = []
neighbors = []
# 遍历文件夹  
for folder in folder_path:
    dir_path = os.path.join(folder,'kmer_k16')
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            if filename.endswith("overlap_stat.tsv"): 
                with open(os.path.join(dir_path, filename), 'r') as file:  
                    for lines in file:
                        line = lines.strip().split('\t')
                        if len(line) > 2 :
                            if line[2] == str(6):
                                with open(output_file, 'a') as output: 
                                    output.write(folder + "\t")  # 写入文件名  
                                    output.writelines(lines)  
            if filename.endswith("overlap_sizes.pkl"): 
                with open(dir_path+'/'+filename,'rb') as f:
                    neighbor_overlap_sizes = pickle.load(f)
                    method = re.search(pattern,filename)[0][:-2]
                    print(method)
                    for i in range(len(neighbor_overlap_sizes[:-1])):
                        nested_list = neighbor_overlap_sizes[:-1][:i+1]
                        flatten_list = list(itertools.chain(*nested_list))
                        integral_precision.append((len(flatten_list)-flatten_list.count(0))/len(flatten_list))
                        integral_mean.append(statistics.mean(flatten_list))
                        samples.append(folder)
                        methods.append(method)
                        neighbors.append(i+1)
di = {'sample':samples,'method':methods,'n_neighbors':neighbors,'integral_precision':integral_precision,'integral_mean':integral_mean}
df = pd.DataFrame(di)
df.to_csv(output_file1,sep='\t')
