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

output_file = f"summarize_time_rss_{file_suffix}.csv"

def wall_clock_to_second(wall_clock):
    hours,minutes,seconds = 0,0,0
    if wall_clock.count(':') == 2:  # h:mm:ss 格式
        hours, minutes, seconds = wall_clock.split(':')
    elif wall_clock.count(':') == 1: 
        minutes, seconds = wall_clock.split(':')
    else:
        seconds = wall_clock

    total_seconds = int(int(hours) * 3600 + int(minutes) * 60 + float(seconds))
    return total_seconds

import re
mrss_pattern = r"Maximum resident set size \(kbytes\): (\d+)\n"
wall_clock_pattern=r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)\n"
with open('/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/CHM13/IGK/real_cyclone/kmer_k16/HNSW_Cosine_PCA_500d_IDF_time.log', 'r') as file: 
    for line in file:
        if re.search(wall_clock_pattern,line):
            total_seconds =  wall_clock_to_second(re.search(wall_clock_pattern,line)[1])
        if re.search(mrss_pattern,line):
            mrss = re.search(mrss_pattern,line)[1]        
time = []
rss = []
samples = []
methods = []
# 遍历文件夹  
for folder in folder_path:
    dir_path = os.path.join(folder,'kmer_k16')
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            if filename.endswith("time.log"): 
                with open(os.path.join(dir_path, filename), 'r') as file:  
                    for line in file:
                        if re.search(wall_clock_pattern,line):
                            time.append(wall_clock_to_second(re.search(wall_clock_pattern,line)[1]))
                            samples.append(folder)
                            methods.append(re.match(r"(.+)_time.log",filename)[1])
                        if re.search(mrss_pattern,line):
                            rss.append(re.search(mrss_pattern,line)[1])

di = {'sample':samples,'method':methods,'wall_clock_time':time,'mrss':rss}
df = pd.DataFrame(di)
df.to_csv(output_file,sep='\t')
