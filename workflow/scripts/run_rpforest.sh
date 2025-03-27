#!/bin/bash
leaf_size=(50 50 50)
no_trees=(100 200 500)
i=2
mkdir /home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/CHM13/chr1_248M/real_cyclone/kmer_k16/RPForest_config$i
log="/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/CHM13/chr1_248M/real_cyclone/kmer_k16/RPForest_config$i/RPForest_config$i.log"
/usr/bin/time -v -o "$log" python /home/miaocj/docker_dir/kNN-overlap-finder/workflow/scripts/run_rpforest.py "$i" "${leaf_size[$i-1]}" "${no_trees[$i-1]}"
