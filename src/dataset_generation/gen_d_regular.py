"""Generate reproducible 4-regular train and size-specific OOD datasets."""
import argparse
import pickle
from pathlib import Path
import networkx as nx
import numpy as np

D = 4
NUM_GRAPHS = 500
TRAIN_SIZE_RANGE = (20, 80)
OOD_SIZES = tuple(range(20, 201, 20))
VAL_RATIO = 0.1
TEST_RATIO = 0.2
SEED = 0

def generate_graphs(num_graphs, min_size, max_size, seed):
    rng = np.random.default_rng(seed)
    valid_sizes = [n for n in range(min_size, max_size + 1) if n > D and n * D % 2 == 0]
    return [nx.random_regular_graph(d=D, n=int(rng.choice(valid_sizes)), seed=int(rng.integers(0, 2**32 - 1))) for _ in range(num_graphs)]

def split_graphs(graphs, seed):
    rng = np.random.default_rng(seed)
    shuffled = [graphs[i] for i in rng.permutation(len(graphs))]
    n_test = int(len(graphs) * TEST_RATIO)
    n_val = int(len(graphs) * VAL_RATIO)
    n_train = len(graphs) - n_val - n_test
    return {"train": shuffled[:n_train], "val": shuffled[n_train:n_train+n_val], "test": shuffled[n_train+n_val:]}

def save_dataset(dataset, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(dataset, handle)
    sizes = [g.number_of_nodes() for split in dataset.values() for g in split]
    print(f"Wrote {path}: splits={ {k: len(v) for k, v in dataset.items()} }, sizes={min(sizes)}..{max(sizes)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    save_dataset(split_graphs(generate_graphs(NUM_GRAPHS, *TRAIN_SIZE_RANGE, seed=SEED), seed=SEED), args.output_dir / "d_regular.pkl")
    for n in OOD_SIZES:
        save_dataset(split_graphs(generate_graphs(NUM_GRAPHS, n, n, seed=SEED+n), seed=SEED+n), args.output_dir / f"d_regular_{n}.pkl")

if __name__ == "__main__":
    main()
