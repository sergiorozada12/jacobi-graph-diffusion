import argparse
import os
import pickle
import numpy as np
import networkx as nx
import scipy as sp

N = 300
NUM_GRAPHS = 500
PLANAR_SIZE_RANGE = (10, 20)
TREE_SIZE_RANGE = (20, 80)
SBM_COMMS_RANGE = (2, 2)
SBM_COMMS_SIZE = (10, 40)
EGO_NUM_EGOS_RANGE = (2, 4)
EGO_SIZE_RANGE = (5, 10)
EGO_INTERCONNECT_PROB = 0.01
EGO_INTRACONNECT_PROB = 0.005
PA_SIZE_RANGE = (20, 80)
PA_ATTACH = 3   
D_REGULAR_DEGREE = 4
D_REGULAR_SIZE_RANGE = (20, 80)
D_REGULAR_OOD_SIZES = tuple(range(20, 201, 20))
SEED = 0
VAL_RATIO = 0.1
TEST_RATIO = 0.2
BASE_PATH = "data/"
PATHS = {
    "planar": os.path.join(BASE_PATH, "planar_graphon.pkl"),
    "tree": os.path.join(BASE_PATH, "tree_graphon.pkl"),
    "sbm": os.path.join(BASE_PATH, f"sbm_2comms_graphon.pkl"),
    "ego": os.path.join(BASE_PATH, "ego_graphon.pkl"),
    "pa": os.path.join(BASE_PATH, "pa_graphon.pkl"),
    "d_regular": os.path.join(BASE_PATH, "d_regular.pkl"),
}


def generate_planar_graphs(num_graphs, min_size, max_size, seed):
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(num_graphs):
        n = rng.integers(min_size, max_size + 1)
        points = rng.random((n, 2))
        tri = sp.spatial.Delaunay(points)
        adj = sp.sparse.lil_array((n, n), dtype=np.int32)
        for t in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    adj[t[i], t[j]] = 1
                    adj[t[j], t[i]] = 1
        G = nx.from_scipy_sparse_array(adj)
        graphs.append(G)
    return graphs


def generate_tree_graphs(num_graphs, min_size, max_size, seed):
    def custom_random_tree(n, rng):
        prufer = rng.integers(0, n, size=n - 2)
        degree = np.ones(n, dtype=int)
        for node in prufer:
            degree[node] += 1
        G = nx.Graph()
        for i in range(n):
            G.add_node(i)
        for node in prufer:
            leaf = np.flatnonzero(degree == 1)[0]
            G.add_edge(leaf, node)
            degree[leaf] -= 1
            degree[node] -= 1
        u, v = np.flatnonzero(degree == 1)
        G.add_edge(u, v)
        return G

    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(num_graphs):
        n = rng.integers(min_size, max_size + 1)
        G = custom_random_tree(n, rng)
        graphs.append(G)
    return graphs


def generate_sbm_graphs(num_graphs, min_comms, max_comms, min_comm_size, max_comm_size, seed):
    rng = np.random.default_rng(seed)
    graphs = []
    while len(graphs) < num_graphs:
        num_communities = rng.integers(min_comms, max_comms + 1)
        comm_size = rng.integers(min_comm_size, max_comm_size + 1)
        community_sizes = [comm_size] * num_communities  # All communities have same size

        probs = np.full((num_communities, num_communities), 0.005)
        np.fill_diagonal(probs, 0.4)

        G = nx.stochastic_block_model(community_sizes, probs, seed=rng)
        if nx.is_connected(G):
            graphs.append(G)
    return graphs

def generate_ego_graphs(num_graphs, num_egos_range, ego_size_range, interconnect_prob, intraconnect_prob, seed):
    rng = np.random.default_rng(seed)
    graphs = []

    for _ in range(num_graphs):
        G = nx.Graph()
        node_counter = 0
        ego_centers = []

        num_egos = rng.integers(*num_egos_range)

        for _ in range(num_egos):
            ego_size = rng.integers(*ego_size_range)
            center = node_counter
            ego_centers.append(center)
            nodes = list(range(node_counter, node_counter + ego_size))
            center_neighbors = nodes[1:]

            for n in center_neighbors:
                G.add_edge(center, n)

            for i in range(1, ego_size):
                for j in range(i + 1, ego_size):
                    if rng.random() < 0.2:
                        G.add_edge(nodes[i], nodes[j])

            node_counter += ego_size

        for i in range(num_egos):
            for j in range(i + 1, num_egos):
                ego_i_nodes = list(nx.ego_graph(G, ego_centers[i], radius=1).nodes)
                ego_j_nodes = list(nx.ego_graph(G, ego_centers[j], radius=1).nodes)

                ego_i_neighbors = [n for n in ego_i_nodes if n != ego_centers[i]]
                ego_j_neighbors = [n for n in ego_j_nodes if n != ego_centers[j]]

                for u in ego_i_neighbors:
                    for v in ego_j_neighbors:
                        if i != j and  rng.random() < interconnect_prob:
                            G.add_edge(u, v)
                        elif i == j and rng.random() < intraconnect_prob:
                            G.add_edge(u, v)

        if nx.is_connected(G):
            graphs.append(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            graphs.append(G.subgraph(largest_cc).copy())

    return graphs


def generate_pa_graphs(num_graphs, size_range, attach, seed):
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(num_graphs):
        n_nodes = int(rng.integers(size_range[0], size_range[1] + 1))
        m_attach = int(np.clip(attach, 1, n_nodes - 1))
        G = nx.barabasi_albert_graph(n_nodes, m_attach, seed=int(rng.integers(0, 10_000)))
        graphs.append(G)
    return graphs


def generate_d_regular_graphs(num_graphs, min_size, max_size, degree, seed):
    rng = np.random.default_rng(seed)
    valid_sizes = [
        n
        for n in range(min_size, max_size + 1)
        if n > degree and n * degree % 2 == 0
    ]
    if not valid_sizes:
        raise ValueError(
            f"No valid {degree}-regular graph sizes in [{min_size}, {max_size}]."
        )
    return [
        nx.random_regular_graph(
            d=degree,
            n=int(rng.choice(valid_sizes)),
            seed=int(rng.integers(0, 2**32 - 1)),
        )
        for _ in range(num_graphs)
    ]

def split_graphs(graphs, val_ratio, test_ratio, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(graphs))
    shuffled = [graphs[i] for i in indices]

    n_total = len(shuffled)
    n_test = max(1, int(n_total * test_ratio)) if test_ratio > 0 else 0
    n_val = max(1, int(n_total * val_ratio)) if val_ratio > 0 else 0
    if n_test + n_val >= n_total:
        # Fall back to keeping at least one training graph
        n_test = max(0, min(n_test, n_total - 1))
        n_val = max(0, min(n_val, n_total - 1 - n_test))

    train_end = n_total - (n_val + n_test)
    val_end = train_end + n_val

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def save_graphs(graphs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graphs, f)
    sizes = [graph.number_of_nodes() for split in graphs.values() for graph in split]
    split_sizes = {split: len(items) for split, items in graphs.items()}
    print(f"Wrote {path}: splits={split_sizes}, sizes={min(sizes)}..{max(sizes)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate all supported synthetic graph datasets."
    )
    parser.add_argument("--output-dir", default=BASE_PATH)
    args = parser.parse_args()

    paths = {
        name: os.path.join(args.output_dir, os.path.basename(path))
        for name, path in PATHS.items()
    }

    tree_graphs = generate_tree_graphs(NUM_GRAPHS, *TREE_SIZE_RANGE, seed=SEED + 1)
    sbm_graphs = generate_sbm_graphs(NUM_GRAPHS, *SBM_COMMS_RANGE, *SBM_COMMS_SIZE, seed=SEED + 2)
    pa_graphs = generate_pa_graphs(NUM_GRAPHS, PA_SIZE_RANGE, PA_ATTACH, seed=SEED + 3)
    d_regular_graphs = generate_d_regular_graphs(
        NUM_GRAPHS,
        *D_REGULAR_SIZE_RANGE,
        degree=D_REGULAR_DEGREE,
        seed=SEED,
    )

    tree_dataset = split_graphs(tree_graphs, VAL_RATIO, TEST_RATIO, seed=SEED + 1)
    sbm_dataset = split_graphs(sbm_graphs, VAL_RATIO, TEST_RATIO, seed=SEED + 2)
    pa_dataset = split_graphs(pa_graphs, VAL_RATIO, TEST_RATIO, seed=SEED + 3)
    d_regular_dataset = split_graphs(
        d_regular_graphs, VAL_RATIO, TEST_RATIO, seed=SEED
    )

    save_graphs(tree_dataset, paths["tree"])
    save_graphs(sbm_dataset, paths["sbm"])
    save_graphs(pa_dataset, paths["pa"])
    save_graphs(d_regular_dataset, paths["d_regular"])

    for size in D_REGULAR_OOD_SIZES:
        graphs = generate_d_regular_graphs(
            NUM_GRAPHS,
            size,
            size,
            degree=D_REGULAR_DEGREE,
            seed=SEED + size,
        )
        dataset = split_graphs(graphs, VAL_RATIO, TEST_RATIO, seed=SEED + size)
        save_graphs(dataset, os.path.join(args.output_dir, f"d_regular_{size}.pkl"))


if __name__ == "__main__":
    main()
