#!/usr/bin/env python
# coding: utf-8

import csv
import os
import numpy as np
import argparse
from tqdm import tqdm

# from tqdm import tqdm_notebook as tqdm
from heapq import heappush, heappop
import random
import itertools

random.seed(108)
from hnsw import (
    HNSW,
    heuristic_improved_optimized,
)
from hnsw import l2_distance, heuristic


def brute_force_knn_search(distance_func, k, q, data):
    """
    Return the list of (idx, dist) for k-closest elements to {x} in {data}
    """
    return sorted(
        enumerate(map(lambda x: distance_func(q, x), data)), key=lambda a: a[1]
    )[:k]


def calculate_recall(distance_func, kg, test, groundtruth, k, ef, m):
    if groundtruth is None:
        print("Ground truth not found. Calculating ground truth...")
        groundtruth = [
            [
                idx
                for idx, dist in brute_force_knn_search(
                    distance_func, k, query, kg.data
                )
            ]
            for query in tqdm(test)
        ]

    print("Calculating recall...")
    recalls = []
    total_calc = 0
    for query, true_neighbors in tqdm(zip(test, groundtruth), total=len(test)):
        true_neighbors = true_neighbors[:k]  # Use only the top k ground truth neighbors
        entry_points = random.sample(range(len(kg.data)), m)
        observed = [
            neighbor
            for neighbor, dist in kg.search(q=query, k=k, ef=ef, return_observed=True)
        ]
        total_calc = total_calc + len(observed)
        results = observed[:k]
        intersection = len(set(true_neighbors).intersection(set(results)))
        recall = intersection / k
        recalls.append(recall)

    return np.mean(recalls), total_calc / len(test)


def read_fvecs(filename):
    with open(filename, "rb") as f:
        while True:
            vec_size = np.fromfile(f, dtype=np.int32, count=1)
            if vec_size.size == 0:
                break
            vec = np.fromfile(f, dtype=np.float32, count=vec_size[0])
            yield vec


def read_ivecs(filename):
    with open(filename, "rb") as f:
        while True:
            vec_size = np.fromfile(f, dtype=np.int32, count=1)
            if vec_size.size == 0:
                break
            vec = np.fromfile(f, dtype=np.int32, count=vec_size[0])
            yield vec


def load_sift_dataset():
    train_file = "datasets/siftsmall/siftsmall_base.fvecs"
    test_file = "datasets/siftsmall/siftsmall_query.fvecs"
    groundtruth_file = "datasets/siftsmall/siftsmall_groundtruth.ivecs"

    train_data = np.array(list(read_fvecs(train_file)))
    test_data = np.array(list(read_fvecs(test_file)))
    groundtruth_data = np.array(list(read_ivecs(groundtruth_file)))

    return train_data, test_data, groundtruth_data


def load_sift1m_dataset():
    train_file = "datasets/sift/sift_base.fvecs"
    test_file = "datasets/sift/sift_query.fvecs"
    groundtruth_file = "datasets/sift/sift_groundtruth.ivecs"

    train_data = np.array(list(read_fvecs(train_file)))
    test_data = np.array(list(read_fvecs(test_file)))
    groundtruth_data = np.array(list(read_ivecs(groundtruth_file)))

    return train_data, test_data, groundtruth_data


def generate_synthetic_data(dim, n, nq):
    train_data = np.random.random((n, dim)).astype(np.float32)
    test_data = np.random.random((nq, dim)).astype(np.float32)
    return train_data, test_data


def add_experiment_result(result_file, result):
    """
    Appends an experiment result to the specified CSV file.

    :param result_file: Path to the CSV file.
    :param result: List of values representing the experiment result.
    """
    if not os.path.isfile(result_file):
        with open(result_file, "w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow(["method", "m", "m0", "ef", "recall", "avg calc"])

    with open(result_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(result)


def main():
    parser = argparse.ArgumentParser(
        description="Test recall of beam search method with KGraph."
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "sift", "sift1m"],
        default="synthetic",
        help="Choose the dataset to use: 'synthetic' or 'sift'.",
    )
    parser.add_argument(
        "--K", type=int, default=5, help="The size of the neighbourhood"
    )
    parser.add_argument("--M", type=int, default=50, help="Avg number of neighbors")
    parser.add_argument("--M0", type=int, default=50, help="Avg number of neighbors")
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Dimensionality of synthetic data (ignored for SIFT).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of training points for synthetic data (ignored for SIFT).",
    )
    parser.add_argument(
        "--nq",
        type=int,
        default=50,
        help="Number of query points for synthetic data (ignored for SIFT).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors to search in the test stage",
    )
    parser.add_argument(
        "--ef", type=int, default=10, help="Size of the beam for beam search."
    )
    parser.add_argument(
        "--m", type=int, default=3, help="Number of random entry points."
    )
    parser.add_argument(
        "--nc",
        choices=["heuristic", "optimized"],
        default="heuristic",
        help="Choose the method of neighborhood construction",
    )

    args = parser.parse_args()

    if args.dataset == "sift":
        print("Loading SIFT dataset...")
        train_data, test_data, groundtruth_data = load_sift_dataset()
    elif args.dataset == "sift1m":
        print("Loading SIFT1m dataset...")
        train_data, test_data, groundtruth_data = load_sift1m_dataset()
    else:
        print(f"Generating synthetic dataset with {args.dim}-dimensional space...")
        train_data, test_data = generate_synthetic_data(args.dim, args.n, args.nq)
        groundtruth_data = None

    if args.nc == "optimized":
        nc_method = heuristic_improved_optimized
    else:
        nc_method = heuristic

    # Create HNSW

    hnsw = HNSW(
        distance_func=l2_distance,
        m=args.M,
        m0=args.M0,
        ef=args.ef,
        ef_construction=30,
        neighborhood_construction=nc_method,
    )
    # Add data to HNSW
    for x in tqdm(train_data):
        hnsw.add(x)

    # Calculate recall
    recall, avg_cal = calculate_recall(
        l2_distance, hnsw, test_data, groundtruth_data, k=args.k, ef=args.ef, m=args.m
    )
    print(f"Recall: {recall}, avg calc: {avg_cal}")

    add_experiment_result(
        "test_hnsw_results.csv", [args.nc, args.M, args.M0, args.ef, recall, avg_cal]
    )


if __name__ == "__main__":
    main()
