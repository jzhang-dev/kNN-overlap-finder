#!/usr/bin/env bash

/home/zhangjiayuan/Software/DRPT/DRPT-main/cpp/build/bin/drpt  \
    -input /home/zhangjiayuan/workspace/kNN-overlap-finder/notebooks/random_projection_forest/data/embedding.fbin.npy \
    -output /home/zhangjiayuan/workspace/kNN-overlap-finder/notebooks/random_projection_forest/data/ \
    -data-set-size 12106 \
    -dimension 100 \
    -ntrees 16 \
    -nn 20 \
    -locality 0 \
    -file_format 1
