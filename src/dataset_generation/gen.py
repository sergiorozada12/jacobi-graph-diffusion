import os
import pickle
import numpy as np
import networkx as nx
import scipy as sp


NUM_GRAPHS = 500
PLANAR_SIZE_RANGE = (10, 20)
TREE_SIZE_RANGE = (10, 20)
SBM_COMMS_RANGE = (2, 2)
SBM_COMMS_SIZE = (10, 20)
EGO_NUM_EGOS_RANGE = (2, 4)
EGO_SIZE_RANGE = (5, 10)
EGO_INTERCONNECT_PROB = 0.01
EGO_INTRACONNECT_PROB = 0.005
SEED = 0
BASE_PATH = "data/"
PATHS = {
    "planar": os.path.join(BASE_PATH, "planar.pkl"),
    "tree": os.path.join(BASE_PATH, "tree.pkl"),
    "sbm": os.path.join(BASE_PATH, "sbm.pkl"),
    "ego": os.path.join(BASE_PATH, "ego.pkl"),
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

def save_graphs(graphs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graphs, f)


if __name__ == "__main__":
    planar_graphs = generate_planar_graphs(NUM_GRAPHS, *PLANAR_SIZE_RANGE, seed=SEED)
    tree_graphs = generate_tree_graphs(NUM_GRAPHS, *TREE_SIZE_RANGE, seed=SEED + 1)
    sbm_graphs = generate_sbm_graphs(NUM_GRAPHS, *SBM_COMMS_RANGE, *SBM_COMMS_SIZE, seed=SEED + 2)
    ego_graphs = generate_ego_graphs(NUM_GRAPHS, EGO_NUM_EGOS_RANGE, EGO_SIZE_RANGE, EGO_INTERCONNECT_PROB, EGO_INTRACONNECT_PROB, seed=SEED)

    save_graphs(planar_graphs, PATHS["planar"])
    save_graphs(tree_graphs, PATHS["tree"])
    save_graphs(sbm_graphs, PATHS["sbm"])
    save_graphs(ego_graphs, PATHS["ego"])
