import os
import sys
import pandas as pd
sys.path.append('/home/miaocj/docker_dir/kNN-overlap-finder/scripts')
from accelerate import parse_fasta
error_rate = []
length = []
reads_num = []
reads_mean_length = []
depth = []

dir_path = "/home/miaocj/docker_dir/kNN-overlap-finder/data/regional_reads/rice/chr1_43M"
for dirname in os.listdir(dir_path):
    if 'pbsim' in dirname and dirname.endswith('dep'):
        print(dirname.split('_')[-3],dirname.split('_')[-2][:-1],dirname.split('_')[-1][:-3])
        lengths = []
        fasta_path = '/'.join([dir_path,dirname,'reads.fasta.gz'])
        for seq in parse_fasta(fasta_path):
            lengths.append(len(seq[1]))
        error_rate.append(dirname.split('_')[-3])
        length.append(dirname.split('_')[-2][:-1])
        depth.append(dirname.split('_')[-1][:-3])
        reads_num.append(len(lengths))
        reads_mean_length.append(sum(lengths)/len(lengths))
di = {'error_rate':error_rate,'read_length':length,'depth':depth,
      'reads_num':reads_num,'reads_mean_length':reads_mean_length}
df = pd.DataFrame(di)
df.to_csv("/home/miaocj/docker_dir/kNN-overlap-finder/data/regional_reads/rice/chr1_43M/simulate_readinfo.tsv",sep='\t')
