#!/bin/bash

m_values=(64 32 16)

ef=64

dataset=datasets/deep10m/base.10M.fbin

result=components-test-results.csv

for m in "${m_values[@]}"
do
    echo "Running with m=$m, ef=$ef, dataset=$dataset, result=$result"
    python3 components_test.py --dataset "$dataset" --m "$m" --ef "$ef" --result "$result"
done
