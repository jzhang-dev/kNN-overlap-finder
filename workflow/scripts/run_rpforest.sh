#!/bin/bash
leaf_size=(50 50 50)
no_trees=(100 200 500)
i=1
samples=('CHM13/HLA/real_pb' 'CHM13/IGK/real_pb' 'CHM13/chr22_51M/real_pb' 'CHM13/chr1_248M/real_pb' )

for sample in "${samples[@]}"
do 
    mkdir "/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation64/${sample}/kmer_k11/RPForest_config$i"
    log="/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation64/${sample}/kmer_k11/RPForest_config$i/RPForest_Cosine_SparseRP_3000d_IDF_time.log"
    /usr/bin/time -v -o "$log" python /home/miaocj/docker_dir/kNN-overlap-finder/workflow/scripts/run_rpforest.py "$i" "${leaf_size[$i-1]}" "${no_trees[$i-1]}" ${sample} &
done
wait