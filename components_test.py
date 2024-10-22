import argparse
import csv
import os
import hnswlib
import numpy as np
import networkx as nx
from tqdm import tqdm

from read_fbin import read_fbin


def calculate_connected_components(dataset, m, ef):
    print(f"Loading dataset from {dataset}")
    data = read_fbin(dataset)

    num_elements = data.shape[0]
    dim = data.shape[1]
    M = m
    ef_construction = ef

    print(
        f"Initializing HNSW graph with parameters M={M}, ef_construction={ef_construction}, elements={num_elements}, dimension={dim}"
    )

    hnsw_index = hnswlib.Index(space="l2", dim=dim)
    hnsw_index.init_index(
        max_elements=num_elements, ef_construction=ef_construction, M=M
    )

    print(f"Adding nodes")

    hnsw_index.add_items(data)

    print(f"Adding edges")

    labels, distances = hnsw_index.knn_query(data, k=M)

    print(f"Calculating number of components")

    del data
    del hnsw_index

    G = nx.Graph()

    for idx, neighbors in enumerate(labels):
        for neighbor in neighbors:
            if idx != neighbor:
                G.add_edge(idx, neighbor)

    connected_components = nx.number_connected_components(G)

    print(f"Number of connected components: {connected_components}")

    return (num_elements, dim, connected_components)


def add_experiment_result(result_file, result):
    if not os.path.isfile(result_file):
        with open(result_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["m", "ef", "num elements", "dimension", "connected components"]
            )

    with open(result_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(result)


def main():
    parser = argparse.ArgumentParser(
        description="Building HNSW graph and calculating connected components"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/deep10m/base.10M.fbin",
        help="Path to dataset file in .fvecs format",
    )
    parser.add_argument(
        "--result",
        type=str,
        default="components-test-results.csv",
        help="Path to the CSV file for results",
    )

    parser.add_argument(
        "--m", type=int, default=16, help="Number of neighbors for HNSW graph (M)"
    )

    parser.add_argument(
        "--ef",
        type=int,
        default=64,
        help="Parameter for constructing HNSW graph (ef_construction)",
    )

    args = parser.parse_args()

    num_elements, dim, connected_components = calculate_connected_components(
        args.dataset, args.m, args.ef
    )

    add_experiment_result(
        args.result, [args.m, args.ef, num_elements, dim, connected_components]
    )


if __name__ == "__main__":
    main()
