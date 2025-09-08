find data/minimap2 -type f -name "minimap2.log" | while read src_file; do
    # 提取可变路径部分（如 ara/chr1_30M/filter3_real_ONT）
    variable_path=$(dirname "$src_file" | sed 's|data/minimap2/||')
    
    # 构建目标路径
    dest_dir="data/evaluation64/${variable_path}/kmer_k11/"
    mkdir -p "$dest_dir"
    
    # 移动并重命名文件
    mv "$src_file" "${dest_dir}/Minimap2_time.log"
done