#!/bin/bash
leaf_size=(50 50 50)
no_trees=(100 200 500)
i=1
samples=('CHM13/chr1_248M/filter3_real_cyclone')

for sample in "${samples[@]}"
do 
    mkdir "/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation64/${sample}/kmer_k21/ann_RPForest_config$i"
    log="/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation64/${sample}/kmer_k21/ann_RPForest_config$i/RPForest_Cosine_SparseRP_2000d_IDF_time.log"
    /usr/bin/time -v -o "$log" python /home/miaocj/docker_dir/kNN-overlap-finder/workflow/scripts/run_rpforest.py "$i" "${leaf_size[$i-1]}" "${no_trees[$i-1]}" ${sample} &
done
wait