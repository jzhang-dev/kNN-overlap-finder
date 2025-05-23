find data/mhap -type f -name "mhap_time.log" | while read src_file; do
    # 提取可变路径部分（如 ara/chr1_30M/filter3_real_ONT）
    variable_path=$(dirname "$src_file" | sed 's|data/mhap/||')
    
    # 构建目标路径
    dest_dir="data/evaluation64/${variable_path}/"
    
    # 移动并重命名文件
    mv "$src_file" "${dest_dir}/MHAP_time.log"
done