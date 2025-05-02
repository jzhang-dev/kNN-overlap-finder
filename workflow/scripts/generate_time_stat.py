import pandas as pd
import pickle, sys, itertools,re
import statistics
import pickle,os,json


mrss_pattern = r"Maximum resident set size \(kbytes\): (\d+)\n"
wall_clock_pattern=r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)\n"
cpu_time_pattern = r"System time \(seconds\): (.+)\n"

def process_overlap_sizes_file(filename: str):
    pattern  = r'data/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/(.+)_time.log'
    all_mrss,all_clock_time,all_cpu_time,all_tfidf_time,all_dim_time,all_ann_time,samples,regions,platforms,encodes,methods,threads = [],[],[],[],[],[],[],[],[],[],[],[]
    with open(filename) as f:
        thread = re.search(pattern,filename).group(1)
        sample = re.search(pattern,filename).group(2)
        region = re.search(pattern,filename).group(3)
        platform = re.search(pattern,filename).group(4) 
        encode = re.search(pattern,filename).group(5)
        method = re.search(pattern,filename).group(6)
        print(thread,sample,region,platform,encode,method)
        for line in f:
            if re.search(mrss_pattern,line):
                mrss = re.search(mrss_pattern,line)[1]  
                all_mrss.append(mrss) 
            if re.search(cpu_time_pattern,line):
                cpu_time = re.search(cpu_time_pattern,line)[1]  
                all_cpu_time.append(cpu_time) 
            if re.search(wall_clock_pattern,line):
                clock_time = re.search(wall_clock_pattern,line)[1]  
                all_clock_time.append(clock_time) 
        methods.append(method)
        samples.append(sample)
        regions.append(region)
        platforms.append(platform)
        encodes.append(encode)
        threads.append(thread)
    nn_time_file= re.search('(.+)_time.log',filename).group(1) + '_nn_time.json'
    if os.path.exists(nn_time_file):
        with open(nn_time_file) as f:
            time_dict = json.load(f)
            if 'dimension_reduction' in time_dict:
                all_dim_time.append(time_dict['dimension_reduction'])
            else:
                all_dim_time.append(None)
            if 'nearest_neighbors' in time_dict:
                all_ann_time.append(time_dict['nearest_neighbors'])
            else:
                all_ann_time.append(None)
            if 'tfidf' in time_dict:
                all_tfidf_time.append(time_dict['tfidf'])
            else:
                all_tfidf_time.append(None)
    else:
        all_dim_time.append(None)
        all_ann_time.append(None)
        all_tfidf_time.append(None)
    return all_mrss,all_clock_time,all_cpu_time,all_tfidf_time,all_dim_time,all_ann_time,samples,regions,platforms,encodes,methods,threads

filename = sys.argv[1]
if len(sys.argv) == 2:
    prefix_pattern = r'(.+)_time.log'
    df_file = re.search(prefix_pattern,filename).group(1) + '_time_rss.tsv'
else:
    df_file =  sys.argv[2]
# if not os.path.exists(df_file):
all_mrss,all_clock_time,all_cpu_time,all_tfidf_time,all_dim_time,all_ann_time,samples,regions,platforms,encodes,methods,threads= process_overlap_sizes_file(filename)
di = {'thread':threads,'sample':samples,'region':regions,'platform':platforms,'encode':encodes,'method':methods,
    'mrss':all_mrss,'clock_time':all_clock_time,'cpu_time':all_cpu_time,'tfidf_time':all_tfidf_time,'dimension_reduction_time':all_dim_time,'get_neighbors_time':all_ann_time}
df = pd.DataFrame(di)
df.to_csv(df_file,sep='\t',index=False)
# else:
#     print(f'{df_file} exists')

