#!/bin/bash

input_file="input.fasta"
batch_size=200000
output_file_index=$1
total_sequences=17217721
start=$((($output_file_index - 1) * $batch_size + 1))
end=$(($output_file_index * $batch_size))
echo "Start: $start"
echo "End: $end"
input_file="/home/miaocj/docker_dir/kNN-overlap-finder/data/metagenome_reference/GTDB/GTDB_rp/GTDB_database_rp.fa"
output_file="/home/miaocj/docker_dir/kNN-overlap-finder/data/metagenome_reference/GTDB/GTDB_rp/GTDB_rp_${output_file_index}.fa"
if [ -f "$output_file" ]; then
    echo "$output_file exists."
else
    # 调用 seqkit 提取序列
    seqkit range -r "${start}:${end}" "$input_file" > "$output_file"
fi
python /home/miaocj/docker_dir/kNN-overlap-finder/workflow/scripts/metagenome/01_encode_GTDB.py "${output_file_index}" 

