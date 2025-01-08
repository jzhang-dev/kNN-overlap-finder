#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *

import csv
import sys
from intervaltree import Interval, IntervalTree
from collections import defaultdict

def load_bam_info_from_tsv(tsv_file):
    """
    从 TSV 文件读入比对信息，返回一个列表:
    [
        {
            "read_name": str,
            "strand": str,
            "read_length": int,
            "ref_name": str,
            "start": int,
            "end": int
        },
        ...
    ]
    """
    bam_info_list = []
    with open(tsv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            bam_info_list.append({
                "read_name": row["read_name"],
                "strand": row["strand"],
                "read_length": int(row["read_length"]),
                "ref_name": row["reference_name"],
                "start": int(row["reference_start"]),
                "end": int(row["reference_end"])
            })
    return bam_info_list


def select_all_alignments(bam_info_list):
    """
    保留所有比对位置。
    返回字典: {
       read_name: [
           {
               "ref_name": str,
               "start": int,
               "end": int,
               "strand": str
           },
           ...
       ]
    }
    """
    read_dict = defaultdict(list)
    for record in bam_info_list:
        rname = record["read_name"]
        ref = record["ref_name"]
        start = record["start"]
        end = record["end"]
        strand = record["strand"]
        
        read_dict[rname].append({
            "ref_name": ref,
            "start": start,
            "end": end,
            "strand": strand
        })
    return read_dict


def build_interval_trees(read_dict):
    """
    将每条 read (保留所有比对位点) 根据 reference_name 构建 IntervalTree。
    返回: { ref_name: IntervalTree(...) }
    """
    interval_trees = defaultdict(IntervalTree)
    
    for rname, alignments in read_dict.items():
        for info in alignments:
            ref = info["ref_name"]
            start = info["start"]
            end = info["end"]
            
            # Interval 的 data 存储 read_name；后续再从 read_dict 获取其他信息
            interval_trees[ref].add(Interval(start, end, data=rname))
        
    return interval_trees


def overlap_length(a_start, a_end, b_start, b_end):
    """
    计算区间 [a_start, a_end) 与 [b_start, b_end) 的重叠长度。
    """
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def find_top_overlaps(read_dict, interval_trees, top_n=6, overlap_threshold=0):
    """
    对 read_dict 中每条 read，从对应的 interval_trees 中查找与其重叠区间最大的前 top_n 条 read。
    仅保留 overlap_size > overlap_threshold 的结果。
    
    返回一个字典:
    {
        read_name: [ (other_read_name, overlap_len), ... ]
    }
    其中列表按 overlap_len 从大到小排序，最多 top_n 条。
    """
    results = {}
    
    for rname, alignments in read_dict.items():
        overlap_data = defaultdict(int)
        
        for info in alignments:
            ref = info["ref_name"]
            start = info["start"]
            end = info["end"]
            
            # 若该染色体无其他 read，直接记空
            if ref not in interval_trees:
                continue
            
            overlap_intervals = interval_trees[ref].overlap(start, end)
            
            # 计算每个 overlap 的大小，并排除自己
            for iv in overlap_intervals:
                other_rname = iv.data
                if other_rname == rname:
                    continue  # 跳过自己
                o_len = overlap_length(start, end, iv.begin, iv.end)
                
                # 只保留 overlap_size 大于阈值的
                if o_len > overlap_threshold:
                    overlap_data[other_rname] += o_len
        
        # 根据 overlap 长度进行降序排序，并取前 top_n 条
        sorted_overlap_data = sorted(overlap_data.items(), key=lambda x: x[1], reverse=True)
        topN = sorted_overlap_data[:top_n]
        
        results[rname] = topN
    
    return results


def write_overlaps_to_tsv(results, read_dict, output_file):
    """
    将结果以以下形式输出到 TSV 文件中:
    read_a, strand_a, read_b, strand_b, overlap_size
    """
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write("read_a\tstrand_a\tread_b\tstrand_b\toverlap_size\n")
        
        for read_a, top_list in results.items():
            strand_a = read_dict[read_a][0]["strand"]
            for (read_b, o_len) in top_list:
                strand_b = read_dict[read_b][0]["strand"]
                out_f.write(
                    f"{read_a}\t{strand_a}\t{read_b}\t{strand_b}\t{o_len}\n"
                )


def load_read_names(fastq_path):
    """
    从 FASTQ 文件中读取 read name，返回一个集合。
    """
    read_names = set()
    with open(fastq_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("@"):
                read_names.add(line.strip()[1:])
    return read_names





def main(snakemake):
    """
    命令行参数说明:
    python find_top_overlaps.py <bam_info.tsv> <output.tsv> [top_n] [overlap_threshold]
    
    例如:
    python find_top_overlaps.py bam_info.tsv overlaps.tsv 10 50
    
    表示查找每条 read 前 10 条 overlap 最大的 reads, 且 overlap_size 要大于 50。
    """
    tsv_file = snakemake.input['tsv']
    output_file = snakemake.output['tsv']
    min_overlap_size=snakemake.params['min_overlap_size']
    k = snakemake.params['k']
    

    # 1. 读取 TSV 中的 BAM 信息
    bam_info_list = load_bam_info_from_tsv(tsv_file)

    read_names = load_read_names(snakemake.input['fastq'])
    bam_info_list = [record for record in bam_info_list if record["read_name"] in read_names]
    
    # 2. 对于有多个比对位置的同一 read，保留所有比对位置
    read_dict = select_all_alignments(bam_info_list)
    
    # 3. 根据 (ref_name, start, end) 构建 IntervalTrees
    interval_trees = build_interval_trees(read_dict)
    
    # 4. 查找每条 read，在同一染色体上 overlap 最大的前 top_n 条 read，且 overlap_size > overlap_threshold
    results = find_top_overlaps(read_dict, interval_trees, top_n=k, overlap_threshold=min_overlap_size)
    
    # 5. 写出到 TSV
    write_overlaps_to_tsv(results, read_dict, output_file)
    print(f"Overlap results have been written to: {output_file}")


if __name__ == "__main__":
    main(snakemake)