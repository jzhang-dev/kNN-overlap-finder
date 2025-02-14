from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *

from typing import Tuple, Dict
import pandas as pd
import gzip, json
import pysam
import scipy.sparse as sp
from collections import Counter
import numpy as np
import pandas as pd

def get_bam_stat_df(bam_file: str,
                    chromosome:str,
                    start:int,
                    end:int) -> Tuple[pd.DataFrame, Dict[str, str]]:
    seq_dict = {}
    bam = pysam.AlignmentFile(bam_file, "rb")
    df_rows = []
    for i, seg in enumerate(bam.fetch(chromosome,start,end)):
        seq_dict[seg.query_name] = seg.query_sequence
        if seg.is_reverse:
            strand = "-"  # 反向链
        else:
            strand = "+"  # 正向链
        if seg.has_tag("NM"):
            mismatch =  seg.get_tag("NM")
        else:
            mismatch = 0 
        row = dict(
            segment_id=i,
            reference_name=seg.reference_name,
            reference_start=seg.reference_start,
            reference_end=seg.reference_end,
            reference_strand = strand,
            mismatch=mismatch,
            query_name=seg.query_name,
            query_alignment_start=seg.query_alignment_start,
            query_length=seg.query_length,
            query_alignment_length=seg.query_alignment_length,
            mapping_quality=seg.mapping_quality,
        )
        df_rows.append(row)
    df = pd.DataFrame(df_rows)
    return df,seq_dict

def main(snakemake: "SnakemakeContext"):
    chr = snakemake.params['chromosome'] ##Attention, modify the fasta to remove prefix
    start = snakemake.params['start']
    end = snakemake.params['end']
    df,seq_dict = get_bam_stat_df(snakemake.input['bam'],chr,start,end)
    ## calculate metric of measuring the quality of alignment
    df['percentage'] =(df['query_alignment_length']-df['mismatch'])/df['query_length']

    ## only keep one best alignment for each read
    max_indices = df.groupby('query_name')['percentage'].idxmax()
    top_hits = df.loc[max_indices]

    ## drop these reads whose best alignment' percentage < 90%, others save in tsv file
    meta_df = top_hits[top_hits.percentage>=0.5]
    info_df = meta_df.loc[:,['query_name','query_length','reference_strand','reference_start','reference_end']]
    info_df.rename(columns={"query_name": "read_name", "query_length": "read_length"}, inplace=True)
    info_df.to_csv(snakemake.output['tsv'], sep='\t', index=False)

    pass_reads = info_df.read_name.tolist()
    pass_dict = {k:seq_dict[k] for k in pass_reads}
    ##stastic if these bad reads have more than one alignment in this region
    all_reads_num = len(top_hits['query_name'].tolist())
    bad_query = top_hits[top_hits.percentage<0.5]['query_name'].tolist()
    bad_query_align = df[df.query_name.isin(bad_query)].groupby('query_name').size().reset_index()
    bad_query_num = bad_query_align.shape[0]
    bad_in_all = bad_query_num/all_reads_num
    more_than_one_align_reads = bad_query_align[bad_query_align.loc[:,0]>1].shape[0]
    per = more_than_one_align_reads/bad_query_num
    print(f'There are {bad_query_num}({bad_in_all:.2%}) reads do not have >90% percentage alignments and {more_than_one_align_reads}({per:.2%}) of them have more than one alignment.')

    ## save good reads to fasta file for following process
    with gzip.open(snakemake.output['fasta'], "wt") as fasta_out:
        for seq_name,sequence in pass_dict.items():
            fasta_out.write(f">{seq_name}\n")  
            fasta_out.write(f"{sequence}\n")  

if __name__ == "__main__":
    main(snakemake)