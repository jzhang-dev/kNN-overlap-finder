import pandas as pd
import pickle, sys, itertools,re
import statistics
import pickle,os

def process_overlap_sizes_file(filename: str):
    pattern  = r'data/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/(.+)_overlap_sizes.pkl'
    integral_precision,integral_mean,samples,regions,platforms,encodes,methods,threads,neighbors = [],[],[],[],[],[],[],[],[]
    with open(filename,'rb') as f:
        neighbor_overlap_sizes = pickle.load(f)
        thread = re.search(pattern,filename).group(1)
        sample = re.search(pattern,filename).group(2)
        region = re.search(pattern,filename).group(3)
        platform = re.search(pattern,filename).group(4)
        encode = re.search(pattern,filename).group(5)
        method = re.search(pattern,filename).group(6)
        print(thread,sample,region,platform,encode,method)
        for i in range(len(neighbor_overlap_sizes[:-1])):
            nested_list = neighbor_overlap_sizes[:-1][:i+1]
            flatten_list = list(itertools.chain(*nested_list))
            integral_precision.append((len(flatten_list)-flatten_list.count(0))/len(flatten_list))
            integral_mean.append(statistics.mean(flatten_list))
            methods.append(method)
            samples.append(sample)
            regions.append(region)
            platforms.append(platform)
            encodes.append(encode)
            threads.append(thread)
            neighbors.append(i+1)
    return integral_precision,integral_mean,threads,samples,regions,platforms,encodes,methods,neighbors

filename = sys.argv[1]
if len(sys.argv) == 2:
    prefix_pattern = r'(.+)_overlap_sizes'
    df_file = re.search(prefix_pattern,filename).group(1) + '_integral_stat.tsv'
else:
    df_file =  sys.argv[2]
if not os.path.exists(df_file):
    integral_precision,integral_mean,threads,samples,regions,platforms,encodes,methods,neighbors = process_overlap_sizes_file(filename)
    di = {'thread':threads,'sample':samples,'region':regions,'platform':platforms,'encode':encodes,'method':methods,
        'n_neighbors':[int(x) for x in neighbors],'integral_precision':[float(x) for x in integral_precision],'integral_mean':[float(x) for x in integral_mean]}
    df = pd.DataFrame(di)
    df.to_csv(df_file,sep='\t',index=False)
else:
    print(f'{df_file} exists')