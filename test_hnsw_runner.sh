#!/bin/bash

m_values=(32 16 8)
ef=64
dataset=sift
nc_values=("optimized" "heuristic")

for m in "${m_values[@]}"
do
    m0=$((2 * m))
    for nc in "${nc_values[@]}"
    do
        echo "Running with M=$m, M0=$m0, ef=$ef, dataset=$dataset, nc=$nc"
        python3 test-hnsw.py --dataset "$dataset" --M "$m" --ef "$ef" --M0 "$m0" --nc "$nc"
    done
done
