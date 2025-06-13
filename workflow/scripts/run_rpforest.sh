#!/bin/bash
leaf_size=(50)
no_trees=(100)
i=1
samples=('CHM13/HLA/filter3_real_wy' 'CHM13/HLA/filter3_real_ONT_R10' 'CHM13/HLA/filter3_real_pb')

for sample in "${samples[@]}"
do 
    mkdir -p "/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation64/${sample}/hash_k21/ann_RPForest_config$i"
    log="/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation64/${sample}/hash_k21/ann_RPForest_config$i/RPForest_Cosine_SparseRP_1000d_IDF_time.log"
    /usr/bin/time -v -o "$log" python /home/miaocj/docker_dir/kNN-overlap-finder/workflow/scripts/run_rpforest.py "$i" "${leaf_size[$i-1]}" "${no_trees[$i-1]}" ${sample} & 
done
wait