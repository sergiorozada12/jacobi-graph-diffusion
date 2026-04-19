###############################################################################
#
# Adapted from https://github.com/lrjconan/GRAN/ which in turn is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
import graph_tool.all as gt

##Navigate to the ./util/orca directory and compile orca.cpp
# g++ -O2 -std=c++11 -o orca orca.cpp
import os
import copy
import signal
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import subprocess as sp
import concurrent.futures
import powerlaw
from collections import Counter

from typing import Any, Dict, List, Optional

import pygsp as pg
import secrets
from string import ascii_uppercase, digits
from datetime import datetime
from scipy.linalg import eigvalsh
from scipy.stats import chi2, ks_2samp, wasserstein_distance
from src.metrics.utils import (
    compute_mmd,
    gaussian_emd,
    gaussian,
    emd,
    gaussian_tv,
)
from src.utils import adjs_to_graphs
import wandb

from src.metrics.abstract import compute_ratios

PRINT_TIME = False
__all__ = [
    "degree_stats",
    "clustering_stats",
    "orbit_stats_all",
    "spectral_stats",
    "eval_acc_lobster_graph",
]

PA_ALPHA_RANGE = (2.4, 3.6)
PA_HUB_SCALING = 0.5
PA_SIGNIFICANCE_LEVEL = 0.01

# Define a timeout handler
def handler(signum, frame):
    raise TimeoutError


# Set the signal handler for the alarm
signal.signal(signal.SIGALRM, handler)


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True, compute_emd=False):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


###############################################################################


def spectral_worker(G, n_eigvals=-1):
    # eigs = nx.laplacian_spectrum(G)
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    except:
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1 : n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def get_spectral_pmf(eigs, max_eig):
    spectral_pmf, _ = np.histogram(
        np.clip(eigs, 0, max_eig), bins=200, range=(-1e-5, max_eig), density=False
    )
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def eigval_stats(
    eig_ref_list, eig_pred_list, max_eig=20, is_parallel=True, compute_emd=False
):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_pmf,
                eig_ref_list,
                [max_eig for i in range(len(eig_ref_list))],
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_pmf,
                eig_pred_list,
                [max_eig for i in range(len(eig_ref_list))],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eig_ref_list)):
            spectral_temp = get_spectral_pmf(eig_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(eig_pred_list)):
            spectral_temp = get_spectral_pmf(eig_pred_list[i])
            sample_pred.append(spectral_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing eig mmd: ", elapsed)
    return mmd_dist


def eigh_worker(G):
    L = nx.normalized_laplacian_matrix(G).todense()
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        eigvals = np.zeros(L[0, :].shape)
        eigvecs = np.zeros(L.shape)
    return (eigvals, eigvecs)


def compute_list_eigh(graph_list, is_parallel=False):
    eigval_list = []
    eigvec_list = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for e_U in executor.map(eigh_worker, graph_list):
                eigval_list.append(e_U[0])
                eigvec_list.append(e_U[1])
    else:
        for i in range(len(graph_list)):
            e_U = eigh_worker(graph_list[i])
            eigval_list.append(e_U[0])
            eigvec_list.append(e_U[1])
    return eigval_list, eigvec_list


def get_spectral_filter_worker(eigvec, eigval, filters, bound=1.4):
    ges = filters.evaluate(eigval)
    linop = []
    for ge in ges:
        linop.append(eigvec @ np.diag(ge) @ eigvec.T)
    linop = np.array(linop)
    norm_filt = np.sum(linop**2, axis=2)
    hist_range = [0, bound]
    hist = np.array(
        [np.histogram(x, range=hist_range, bins=100)[0] for x in norm_filt]
    )  # NOTE: change number of bins
    return hist.flatten()


def spectral_filter_stats(
    eigvec_ref_list,
    eigval_ref_list,
    eigvec_pred_list,
    eigval_pred_list,
    is_parallel=False,
    compute_emd=False,
):
    """Compute the distance between the eigvector sets.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    prev = datetime.now()

    class DMG(object):
        """Dummy Normalized Graph"""

        lmax = 2

    n_filters = 12
    filters = pg.filters.Abspline(DMG, n_filters)
    bound = np.max(filters.evaluate(np.arange(0, 2, 0.01)))
    sample_ref = []
    sample_pred = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_filter_worker,
                eigvec_ref_list,
                eigval_ref_list,
                [filters for i in range(len(eigval_ref_list))],
                [bound for i in range(len(eigval_ref_list))],
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_filter_worker,
                eigvec_pred_list,
                eigval_pred_list,
                [filters for i in range(len(eigval_pred_list))],
                [bound for i in range(len(eigval_pred_list))],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eigval_ref_list)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    eigvec_ref_list[i], eigval_ref_list[i], filters, bound
                )
                sample_ref.append(spectral_temp)
            except:
                pass
        for i in range(len(eigval_pred_list)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    eigvec_pred_list[i], eigval_pred_list[i], filters, bound
                )
                sample_pred.append(spectral_temp)
            except:
                pass

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing spectral filter stats: ", elapsed)
    return mmd_dist


def spectral_stats(
    graph_ref_list, graph_pred_list, is_parallel=True, n_eigvals=-1, compute_emd=False
):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                spectral_worker, graph_ref_list, [n_eigvals for i in graph_ref_list]
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                spectral_worker,
                graph_pred_list_remove_empty,
                [n_eigvals for i in graph_pred_list_remove_empty],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i], n_eigvals)
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i], n_eigvals)
            sample_pred.append(spectral_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


###############################################################################


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
    )
    return hist


def clustering_stats(
    graph_ref_list, graph_pred_list, bins=100, is_parallel=True, compute_emd=False
):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]
            ):
                sample_pred.append(clustering_hist)

        # check non-zero elements in hist
        # total = 0
        # for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        # print(total)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values()
            )
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd, sigma=1.0 / 10)
        mmd_dist = compute_mmd(
            sample_ref,
            sample_pred,
            kernel=gaussian_emd,
            sigma=1.0 / 10,
            distance_scaling=bins,
        )
    else:
        mmd_dist = compute_mmd(
            sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10
        )

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing clustering mmd: ", elapsed)
    return mmd_dist


# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    "3path": [1, 2],
    "4cycle": [8],
}
COUNT_START_STR = "orbit counts:"


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for u, v in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    # tmp_fname = f'analysis/orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = f'orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), tmp_fname)
    # print(tmp_fname, flush=True)
    f = open(tmp_fname, "w")
    f.write(str(graph.number_of_nodes()) + " " + str(graph.number_of_edges()) + "\n")
    for u, v in edge_list_reindexed(graph):
        f.write(str(u) + " " + str(v) + "\n")
    f.close()
    output = sp.check_output(
        [
            str(os.path.join(os.path.dirname(os.path.realpath(__file__)), "orca/orca")),
            "node",
            "4",
            tmp_fname,
            "std",
        ]
    )
    output = output.decode("utf8").strip()
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR) + 2
    output = output[idx:]
    node_orbit_counts = np.array(
        [
            list(map(int, node_cnts.strip().split(" ")))
            for node_cnts in output.strip("\n").split("\n")
        ]
    )

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def motif_stats(
    graph_ref_list,
    graph_pred_list,
    motif_type="4cycle",
    ground_truth_match=None,
    bins=100,
    compute_emd=False,
):
    # graph motif counts (int for each graph)
    # normalized by graph size
    total_counts_ref = []
    total_counts_pred = []

    num_matches_ref = []
    num_matches_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]
    indices = motif_to_indices[motif_type]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_ref.append(match_cnt / G.number_of_nodes())

        # hist, _ = np.histogram(
        #        motif_counts, bins=bins, density=False)
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_ref.append(motif_temp)

    for G in graph_pred_list_remove_empty:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_pred.append(match_cnt / G.number_of_nodes())

        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_pred.append(motif_temp)

    total_counts_ref = np.array(total_counts_ref)[:, None]
    total_counts_pred = np.array(total_counts_pred)[:, None]

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=emd, is_hist=False)
        mmd_dist = compute_mmd(
            total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False
        )
    else:
        mmd_dist = compute_mmd(
            total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False
        )
    return mmd_dist


def orbit_stats_all(graph_ref_list, graph_pred_list, compute_emd=False):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)

    # mmd_dist = compute_mmd(
    #     total_counts_ref,
    #     total_counts_pred,
    #     kernel=gaussian,
    #     is_hist=False,
    #     sigma=30.0)

    # mmd_dist = compute_mmd(
    #         total_counts_ref,
    #         total_counts_pred,
    #         kernel=gaussian_tv,
    #         is_hist=False,
    #         sigma=30.0)

    if compute_emd:
        # mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=emd, sigma=30.0)
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(
            total_counts_ref,
            total_counts_pred,
            kernel=gaussian,
            is_hist=False,
            sigma=30.0,
        )
    else:
        mmd_dist = compute_mmd(
            total_counts_ref,
            total_counts_pred,
            kernel=gaussian_tv,
            is_hist=False,
            sigma=30.0,
        )
    return mmd_dist


def eval_acc_lobster_graph(G_list):
    G_list = [copy.deepcopy(gg) for gg in G_list]
    count = 0
    for gg in G_list:
        if is_lobster_graph(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_tree_graph(G_list):
    count = 0
    for gg in G_list:
        if nx.is_tree(gg):
            count += 1
    return count / float(len(G_list))

def eval_tree_structure_metrics(G_list, train_size_range=None):
    train_range_min, train_range_max = (None, None)
    if train_size_range is not None:
        train_range_min, train_range_max = train_size_range

    if len(G_list) == 0:
        return {
            "tree_acc": 0.0,
            "forest_acc": 0.0,
            "connected_acc": 0.0,
            "mean_num_components": 0.0,
            "mean_lcc_fraction": 0.0,
            "tree_train_range_node_mass_fraction": 0.0,
        }

    tree_count = 0
    forest_count = 0
    connected_count = 0
    num_components_sum = 0.0
    lcc_ratio_sum = 0.0
    train_range_node_mass_sum = 0.0

    for gg in G_list:
        n_nodes = gg.number_of_nodes()
        if nx.is_tree(gg):
            tree_count += 1
        if nx.is_forest(gg):
            forest_count += 1

        if n_nodes > 0:
            components = list(nx.connected_components(gg))

            # Fraction of nodes that belong to components in the training size range.
            if train_range_min is not None and train_range_max is not None:
                nodes_in_train_range = sum(
                    len(c)
                    for c in components
                    if train_range_min <= len(c) <= train_range_max
                )
                train_range_node_mass_sum += nodes_in_train_range / float(n_nodes)
            else:
                train_range_node_mass_sum += 0.0

            if nx.is_connected(gg):
                connected_count += 1
                num_components_sum += 1.0
                lcc_ratio_sum += 1.0
            else:
                n_components = len(components)
                num_components_sum += float(n_components)
                largest_cc_size = max((len(c) for c in components), default=0)
                lcc_ratio_sum += largest_cc_size / float(n_nodes)
        else:
            num_components_sum += 0.0
            lcc_ratio_sum += 0.0
            train_range_node_mass_sum += 0.0

    denom = float(len(G_list))
    return {
        "tree_acc": tree_count / denom,
        "forest_acc": forest_count / denom,
        "connected_acc": connected_count / denom,
        "mean_num_components": num_components_sum / denom,
        "mean_lcc_fraction": lcc_ratio_sum / denom,
        "tree_train_range_node_mass_fraction": train_range_node_mass_sum / denom,
    }


def eval_acc_grid_graph(G_list, grid_start=10, grid_end=20):
    count = 0
    for gg in G_list:
        if is_grid_graph(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_sbm_graph(
    G_list,
    p_intra=0.4,    # 0.3 for the SPECTRE, 0.4 for the 2-COMMS
    p_inter=0.005,
    strict=True,
    refinement_steps=100,
    seed=0,
    is_parallel=True,
):
    count = 0.0
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for prob in executor.map(
                is_sbm_graph,
                [gg for gg in G_list],
                [p_intra for i in range(len(G_list))],
                [p_inter for i in range(len(G_list))],
                [strict for i in range(len(G_list))],
                [refinement_steps for i in range(len(G_list))],
                [seed for i in range(len(G_list))],
            ):
                count += prob
    else:
        for gg in G_list:
            count += is_sbm_graph(
                gg,
                p_intra=p_intra,
                p_inter=p_inter,
                strict=strict,
                refinement_steps=refinement_steps,
                seed=seed,
            )
    return count / float(len(G_list))


def eval_acc_planar_graph(G_list):
    count = 0
    for gg in G_list:
        if is_planar_graph(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_pa_graph(G_list, debug=False):
    """Simple PA accuracy; set debug=True for detailed diagnostics."""
    if len(G_list) == 0:
        return 0.0

    passed = 0
    failure_reasons = Counter()
    failure_examples = {}
    size_total = Counter()
    size_valid = Counter()
    max_examples_per_reason = 3

    for gg in G_list:
        n_nodes = gg.number_of_nodes()
        # Keep pa_acc on the same PA validity interface as sampling validity.
        is_valid, reason = is_pa_graph_with_reason(
            gg,
            alpha_range=PA_ALPHA_RANGE,
            hub_scaling=PA_HUB_SCALING,
            significance_level=PA_SIGNIFICANCE_LEVEL,
        )
        size_total[n_nodes] += 1
        if is_valid:
            passed += 1
            size_valid[n_nodes] += 1
        else:
            failure_reasons[reason] += 1
            if debug and len(failure_examples.get(reason, [])) < max_examples_per_reason:
                failure_examples.setdefault(reason, []).append(
                    _summarize_graph_basic_stats(gg)
                )

    if debug:
        _print_pa_debug_summary(
            total_graphs=len(G_list),
            passed=passed,
            failure_reasons=failure_reasons,
            size_total=size_total,
            size_valid=size_valid,
            failure_examples=failure_examples,
        )
    return passed / float(len(G_list))


def is_planar_graph(G):
    return nx.is_connected(G) and nx.check_planarity(G)[0]


def is_lobster_graph(G):
    """
    Check a given graph is a lobster graph or not

    Removing leaf nodes twice:

    lobster -> caterpillar -> path

    """
    ### Check if G is a tree
    if nx.is_tree(G):
        G = G.copy()
        ### Check if G is a path after removing leaves twice
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        num_nodes = len(G.nodes())
        num_degree_one = [d for n, d in G.degree() if d == 1]
        num_degree_two = [d for n, d in G.degree() if d == 2]

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        else:
            return False
    else:
        return False


def is_grid_graph(G):
    """
    Check if the graph is grid, by comparing with all the real grids with the same node count
    """
    all_grid_file = f"data/all_grids.pt"
    if os.path.isfile(all_grid_file):
        all_grids = torch.load(all_grid_file)
    else:
        all_grids = {}
        for i in range(2, 20):
            for j in range(2, 20):
                G_grid = nx.grid_2d_graph(i, j)
                n_nodes = f"{len(G_grid.nodes())}"
                all_grids[n_nodes] = all_grids.get(n_nodes, []) + [G_grid]
        torch.save(all_grids, all_grid_file)

    n_nodes = f"{len(G.nodes())}"
    if n_nodes in all_grids:
        for G_grid in all_grids[n_nodes]:
            if nx.faster_could_be_isomorphic(G, G_grid):
                if nx.is_isomorphic(G, G_grid):
                    return True
        return False
    else:
        return False


def is_sbm_graph(G, p_intra=0.4, p_inter=0.005, strict=True, refinement_steps=100, seed=0):
    """
    Check if how closely given graph matches a SBM with given probabilites by computing mean probability of Wald test statistic for each recovered parameter
    """

    adj = nx.adjacency_matrix(G).toarray()
    idx = adj.nonzero()
    g = gt.Graph()
    g.add_edge_list(np.transpose(idx))
    gt.seed_rng(seed)
    try:
        state = gt.minimize_blockmodel_dl(g)
    except ValueError:
        if strict:
            return False
        else:
            return 0.0

    # Refine using merge-split MCMC
    for i in range(refinement_steps):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    b = state.get_blocks()
    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = e.todense()[:n_blocks, :n_blocks]
    if strict:
        if (
            #(node_counts > 40).sum() > 0
            #or (node_counts < 20).sum() > 0
            n_blocks > 5
            or n_blocks < 2
        ):
            return False

    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = edge_counts / (max_inter_edges + 1e-6)

    W_p_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)

    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p = 1 - chi2.cdf(abs(W), 1)
    p = p.mean()
    if strict:
        return p > 0.9  # p value < 10 %
    else:
        return p

PA_REASON_STEPS = [
    ("not_connected", "1) connected"),
    ("too_few_positive_degrees", "2) enough positive degrees"),
    ("powerlaw_fit_exception", "3) powerlaw fit"),
    ("non_finite_alpha", "4) alpha finite"),
    ("alpha_out_of_range", "5) alpha range"),
    ("non_finite_xmin", "6) xmin finite"),
    ("tail_too_small", "7) tail size"),
    ("distribution_compare_exception", "8) distribution compare"),
    ("non_finite_distribution_stats", "9) compare stats finite"),
    ("exp_better_than_powerlaw_significant", "10) powerlaw vs exponential"),
    ("hub_threshold_not_met", "11) hub threshold"),
]


def _summarize_graph_basic_stats(G):
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "density": 0.0,
            "max_degree": 0,
            "mean_degree": 0.0,
            "connected": False,
        }

    degrees = np.array([d for _, d in G.degree()], dtype=float)
    density = nx.density(G) if n_nodes > 1 else 0.0
    connected = nx.is_connected(G) if n_nodes > 0 else False
    return {
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "density": float(density),
        "max_degree": int(degrees.max()) if degrees.size else 0,
        "mean_degree": float(degrees.mean()) if degrees.size else 0.0,
        "connected": bool(connected),
    }


def _print_pa_failure_examples(failure_examples):
    if not failure_examples:
        print("PA failure examples: none")
        return

    print("PA failure examples (per reason):")
    for reason in sorted(failure_examples.keys()):
        print(f"  - reason={reason}:")
        for idx, stats in enumerate(failure_examples[reason], start=1):
            print(
                "      "
                f"#{idx} "
                f"n={stats['n_nodes']}, "
                f"m={stats['n_edges']}, "
                f"density={stats['density']:.4f}, "
                f"max_deg={stats['max_degree']}, "
                f"mean_deg={stats['mean_degree']:.3f}, "
                f"connected={stats['connected']}"
            )


def _print_pa_debug_summary(
    total_graphs, passed, failure_reasons, size_total, size_valid, failure_examples
):
    print("PA check failure reason counts:")
    for reason, num in failure_reasons.most_common():
        print(f"  - {reason}: {num}")

    reason_to_step = {reason: idx + 1 for idx, (reason, _) in enumerate(PA_REASON_STEPS)}
    fail_at_step = Counter()
    for reason, num in failure_reasons.items():
        if reason in reason_to_step:
            fail_at_step[reason_to_step[reason]] += num

    survived = total_graphs
    print("PA check step funnel:")
    for idx, (_, label) in enumerate(PA_REASON_STEPS, start=1):
        failed_here = fail_at_step[idx]
        fail_ratio = failed_here / float(total_graphs)
        print(
            f"  - {label}: failed={failed_here}/{total_graphs} ({fail_ratio:.3f}), "
            f"survived_before_step={survived}"
        )
        survived -= failed_here
    print(
        f"  - 12) final pass: passed={passed}/{total_graphs} "
        f"({passed / float(total_graphs):.3f})"
    )

    print("PA validity by 10-node size bins:")
    bin_total = Counter()
    bin_valid = Counter()
    for n_nodes, total in size_total.items():
        size_bin = (n_nodes // 10) * 10
        bin_label = f"{size_bin:03d}-{size_bin + 9:03d}"
        bin_total[bin_label] += total
        bin_valid[bin_label] += size_valid[n_nodes]
    for bin_label in sorted(bin_total):
        total = bin_total[bin_label]
        valid = bin_valid[bin_label]
        ratio = valid / float(total) if total else 0.0
        print(f"  - n={bin_label}: valid={valid}/{total} ({ratio:.3f})")

    if len(size_total) >= 2:
        sizes = np.array(sorted(size_total.keys()), dtype=float)
        validity = np.array(
            [size_valid[int(s)] / float(size_total[int(s)]) for s in sizes], dtype=float
        )
        counts = np.array([size_total[int(s)] for s in sizes], dtype=float)

        pearson_r = np.corrcoef(sizes, validity)[0, 1]
        size_mean = np.average(sizes, weights=counts)
        val_mean = np.average(validity, weights=counts)
        cov = np.average((sizes - size_mean) * (validity - val_mean), weights=counts)
        var_size = np.average((sizes - size_mean) ** 2, weights=counts)
        var_val = np.average((validity - val_mean) ** 2, weights=counts)
        weighted_pearson_r = cov / np.sqrt(var_size * var_val + 1e-12)
        size_ranks = np.argsort(np.argsort(sizes)).astype(float)
        val_ranks = np.argsort(np.argsort(validity)).astype(float)
        spearman_r = np.corrcoef(size_ranks, val_ranks)[0, 1]
        slope_per_10_nodes = np.polyfit(sizes, validity, 1)[0] * 10.0

        print("PA size-validity trend summary:")
        print(f"  - pearson_r (size vs validity): {pearson_r:.3f}")
        print(f"  - weighted_pearson_r: {weighted_pearson_r:.3f}")
        print(f"  - spearman_rho: {spearman_r:.3f}")
        print(f"  - linear_slope_per_10_nodes: {slope_per_10_nodes:.3f}")

    _print_pa_failure_examples(failure_examples)

def is_pa_graph(
    G,
    alpha_range=PA_ALPHA_RANGE,
    hub_scaling=PA_HUB_SCALING,
    significance_level=PA_SIGNIFICANCE_LEVEL,
):
    is_valid, _ = is_pa_graph_with_reason(
        G,
        alpha_range=alpha_range,
        hub_scaling=hub_scaling,
        significance_level=significance_level,
    )
    return is_valid


def is_pa_graph_simple(G):
    """Short default check for future use."""
    return is_pa_graph(G, alpha_range=(2.4, 3.6), hub_scaling=0.5, significance_level=0.0)


def is_pa_graph_with_reason(
    G,
    alpha_range=PA_ALPHA_RANGE,  # 2.6, 3.4
    hub_scaling=PA_HUB_SCALING,  # 0.7
    significance_level=PA_SIGNIFICANCE_LEVEL,  # 0.001
    tail_min_count=2,
):
    if not nx.is_connected(G):
        return False, "not_connected"
    degrees = np.array([d for _, d in G.degree()], dtype=int)
    degrees = degrees[degrees > 0]
    if degrees.size < 5:
        return False, "too_few_positive_degrees"
    base_min_degree = int(degrees.min())
    try:
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
    except Exception:
        return False, "powerlaw_fit_exception"
    alpha = getattr(fit.power_law, "alpha", np.nan)
    if not np.isfinite(alpha):
        return False, "non_finite_alpha"
    if not (alpha_range[0] <= alpha <= alpha_range[1]):
        return False, "alpha_out_of_range"
    xmin = getattr(fit.power_law, "xmin", np.nan)
    if not np.isfinite(xmin):
        return False, "non_finite_xmin"
    tail_count = np.sum(degrees >= xmin)
    if tail_count < tail_min_count:
        return False, "tail_too_small"
    try:
        R, p = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
    except Exception:
        return False, "distribution_compare_exception"
    if not np.isfinite(R) or not np.isfinite(p):
        return False, "non_finite_distribution_stats"
    if R < 0 and p < significance_level:
        return False, "exp_better_than_powerlaw_significant"
    max_deg = degrees.max()
    n = G.number_of_nodes()
    if max_deg < hub_scaling * base_min_degree * np.sqrt(n):
        return False, "hub_threshold_not_met"
    return True, "pass"


def eval_fraction_isomorphic(fake_graphs, train_graphs):
    count = 0
    for fake_g in fake_graphs:
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                if nx.is_isomorphic(fake_g, train_g):
                    count += 1
                    break
    return count / float(len(fake_graphs))


def eval_fraction_unique(fake_graphs, precise=False):
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True
        if not fake_g.number_of_nodes() == 0:
            for fake_old in fake_evaluated:
                if precise:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.is_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
                else:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.could_be_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
            if unique:
                fake_evaluated.append(fake_g)

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs

    return frac_unique


def eval_fraction_unique_non_isomorphic_valid(
    fake_graphs, train_graphs, validity_func=(lambda x: True)
):
    count_valid = 0
    count_isomorphic = 0
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True

        for fake_old in fake_evaluated:
            try:
                # Set the alarm for 60 seconds
                signal.alarm(60)
                if nx.is_isomorphic(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break
            except TimeoutError:
                print("Timeout: Skipping this iteration")
                continue
            finally:
                # Disable the alarm
                signal.alarm(0)
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    if nx.is_isomorphic(fake_g, train_g):
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
            if non_isomorphic:
                if validity_func(fake_g):
                    count_valid += 1

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs
    frac_unique_non_isomorphic = (
        float(len(fake_graphs)) - count_non_unique - count_isomorphic
    ) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set
    frac_unique_non_isomorphic_valid = count_valid / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set and are valid
    return frac_unique, frac_unique_non_isomorphic, frac_unique_non_isomorphic_valid


class SpectreSamplingMetrics(nn.Module):
    def __init__(self, datamodule, compute_emd, metrics_list):
        super().__init__()

        self.train_graphs = self.loader_to_nx(datamodule.train_dataloader())
        self.val_graphs = self.loader_to_nx(datamodule.val_dataloader())
        self.test_graphs = self.loader_to_nx(datamodule.test_dataloader())
        self.num_graphs_test = len(self.test_graphs)
        self.num_graphs_val = len(self.val_graphs)
        self.compute_emd = compute_emd
        self.metrics_list = metrics_list

        # Store for wavelet computaiton
        self.val_ref_eigvals, self.val_ref_eigvecs = compute_list_eigh(self.val_graphs)
        self.test_ref_eigvals, self.test_ref_eigvecs = compute_list_eigh(
            self.test_graphs
        )
        self.train_size_range = self._infer_train_size_range()

    def _infer_train_size_range(self):
        node_sizes = [g.number_of_nodes() for g in self.train_graphs if g is not None]
        return (min(node_sizes), max(node_sizes))

    def loader_to_nx(self, loader):
        networkx_graphs = []
        for i, batch in enumerate(loader):
            adjs = batch[1]
            G = adjs_to_graphs(adjs, is_cuda=True)
            networkx_graphs.extend(G)
        return networkx_graphs

    def forward(
        self,
        generated_graphs: list,
        ref_metrics,
        extra_ref_metrics=None,
        local_rank=0,
        test=False,
    ):
        reference_graphs = self.test_graphs if test else self.val_graphs
        if local_rank == 0:
            print(
                f"Computing sampling metrics between {len(generated_graphs)} generated graphs and {len(reference_graphs)}"
                f" test graphs -- emd computation: {self.compute_emd}"
            )
        networkx_graphs = generated_graphs

        to_log = {}
        if "degree" in self.metrics_list:
            if local_rank == 0:
                print("Computing degree stats..")
            degree = degree_stats(
                reference_graphs,
                networkx_graphs,
                is_parallel=True,
                compute_emd=self.compute_emd,
            )
            if wandb.run:
                wandb.run.summary["degree"] = degree
            to_log["degree"] = degree

        if "wavelet" in self.metrics_list:
            if local_rank == 0:
                print("Computing wavelet stats...")

            ref_eigvecs = self.test_ref_eigvecs if test else self.val_ref_eigvecs
            ref_eigvals = self.test_ref_eigvals if test else self.val_ref_eigvals

            pred_graph_eigvals, pred_graph_eigvecs = compute_list_eigh(networkx_graphs)
            wavelet = spectral_filter_stats(
                eigvec_ref_list=ref_eigvecs,
                eigval_ref_list=ref_eigvals,
                eigvec_pred_list=pred_graph_eigvecs,
                eigval_pred_list=pred_graph_eigvals,
                is_parallel=False,
                compute_emd=self.compute_emd,
            )
            to_log["wavelet"] = wavelet
            if wandb.run:
                wandb.run.summary["wavelet"] = wavelet

        if "spectre" in self.metrics_list:
            if local_rank == 0:
                print("Computing spectre stats...")
            spectre = spectral_stats(
                reference_graphs,
                networkx_graphs,
                is_parallel=True,
                n_eigvals=-1,
                compute_emd=self.compute_emd,
            )

            to_log["spectre"] = spectre
            if wandb.run:
                wandb.run.summary["spectre"] = spectre

        if "clustering" in self.metrics_list:
            if local_rank == 0:
                print("Computing clustering stats...")
            clustering = clustering_stats(
                reference_graphs,
                networkx_graphs,
                bins=100,
                is_parallel=True,
                compute_emd=self.compute_emd,
            )
            to_log["clustering"] = clustering
            if wandb.run:
                wandb.run.summary["clustering"] = clustering

        if "motif" in self.metrics_list:
            if local_rank == 0:
                print("Computing motif stats")
            motif = motif_stats(
                reference_graphs,
                networkx_graphs,
                motif_type="4cycle",
                ground_truth_match=None,
                bins=100,
                compute_emd=self.compute_emd,
            )
            to_log["motif"] = motif
            if wandb.run:
                wandb.run.summary["motif"] = motif

        if "orbit" in self.metrics_list:
            if local_rank == 0:
                print("Computing orbit stats...")
            orbit = orbit_stats_all(
                reference_graphs, networkx_graphs, compute_emd=self.compute_emd
            )
            to_log["orbit"] = orbit
            if wandb.run:
                wandb.run.summary["orbit"] = orbit

        if "sbm" in self.metrics_list:
            if local_rank == 0:
                print("Computing accuracy...")
            sbm_acc = eval_acc_sbm_graph(
                networkx_graphs, refinement_steps=100, strict=True, is_parallel=True
            )
            to_log["sbm_acc"] = sbm_acc
            if wandb.run:
                wandb.run.summary["sbm_acc"] = sbm_acc

        if "pa" in self.metrics_list:
            if local_rank == 0:
                print("Computing PA accuracy...")
            pa_acc = eval_acc_pa_graph(networkx_graphs)
            to_log["pa_acc"] = pa_acc
            if wandb.run:
                wandb.run.summary["pa_acc"] = pa_acc

        if "planar" in self.metrics_list:
            if local_rank == 0:
                print("Computing planar accuracy...")
            planar_acc = eval_acc_planar_graph(networkx_graphs)
            to_log["planar_acc"] = planar_acc
            if wandb.run:
                wandb.run.summary["planar_acc"] = planar_acc

        if "tree" in self.metrics_list:
            if local_rank == 0:
                print("Computing tree accuracy...")
            tree_metrics = eval_tree_structure_metrics(
                networkx_graphs, train_size_range=self.train_size_range
            )
            to_log.update(tree_metrics)
            if wandb.run:
                for key, value in tree_metrics.items():
                    wandb.run.summary[key] = value

        if (
            "sbm" in self.metrics_list
            or "planar" in self.metrics_list
            or "tree" in self.metrics_list
            or "pa" in self.metrics_list
        ):
            if local_rank == 0:
                print("Computing all fractions...")
            if "sbm" in self.metrics_list:
                validity_func = is_sbm_graph
            elif "planar" in self.metrics_list:
                validity_func = is_planar_graph
            elif "tree" in self.metrics_list:
                validity_func = nx.is_tree
            elif "pa" in self.metrics_list:
                validity_func = is_pa_graph
            else:
                validity_func = None
            (
                frac_unique,
                frac_unique_non_isomorphic,
                fraction_unique_non_isomorphic_valid,
            ) = eval_fraction_unique_non_isomorphic_valid(
                networkx_graphs,
                self.train_graphs,
                validity_func,
            )
            frac_non_isomorphic = 1.0 - eval_fraction_isomorphic(
                networkx_graphs, self.train_graphs
            )
            to_log.update(
                {
                    "sampling/frac_unique": frac_unique,
                    "sampling/frac_unique_non_iso": frac_unique_non_isomorphic,
                    "sampling/frac_unic_non_iso_valid": fraction_unique_non_isomorphic_valid,
                    "sampling/frac_non_iso": frac_non_isomorphic,
                }
            )

        ratios = compute_ratios(
            gen_metrics=to_log,
            ref_metrics=ref_metrics["test"] if test else ref_metrics["val"],
            metrics_keys=["degree", "clustering", "orbit", "spectre", "wavelet"],
        )
        to_log.update(ratios)
        if extra_ref_metrics:
            for suffix, extra_metrics in extra_ref_metrics.items():
                extra_ratios = compute_ratios(
                    gen_metrics=to_log,
                    ref_metrics=extra_metrics["test"] if test else extra_metrics["val"],
                    metrics_keys=["degree", "clustering", "orbit", "spectre", "wavelet"],
                )
                to_log.update({f"{key}_{suffix}": value for key, value in extra_ratios.items()})

        print("Sampling statistics", to_log)
        if wandb.run:
            wandb.log(to_log)
        return to_log

    def reset(self):
        pass


class Comm20SamplingMetrics(SpectreSamplingMetrics):

    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=True,
            metrics_list=["degree", "clustering", "orbit", "spectre", "wavelet"],
        )


class PlanarSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=False,
            metrics_list=[
                "degree",
                "clustering",
                "orbit",
                "spectre",
                "wavelet",
                "planar",
            ],
        )


class SBMSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=False,
            metrics_list=["degree", "clustering", "orbit", "spectre", "wavelet", "sbm"],
        )


class PASamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=False,
            metrics_list=["degree", "clustering", "orbit", "spectre", "wavelet", "pa"],
        )


class TreeSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=False,
            metrics_list=[
                "degree",
                "clustering",
                "orbit",
                "spectre",
                "wavelet",
                "tree",
            ],
        )


class EgoSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=False,
            metrics_list=["degree", "clustering", "orbit", "spectre", "wavelet"],
        )


class ProteinSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=False,
            metrics_list=["degree", "clustering", "orbit", "spectre", "wavelet"],
        )


class IMDBSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=False,
            metrics_list=["degree", "clustering", "orbit", "spectre", "wavelet"],
        )


class WirelessSamplingMetrics:
    """
    Metrics module for MetroFi experiments.

    Computes two families of metrics:
    1) Edge-wise interference distribution distances (KS and Wasserstein)
       across the 70x70 MAC universe.
    2) Structural metrics (degree, spectre) on subgraphs sampled from generated
       full graphs using the empirical size distribution of the test set.
    """

    def __init__(self, datamodule):
        self.train_graphs = datamodule.train_graphs
        self.val_graphs = datamodule.val_graphs
        self.test_graphs = datamodule.test_graphs

        self.test_size_support, self.test_size_probs = self._graph_size_distribution(self.test_graphs)
        self.rng = np.random.default_rng()

    def reset(self) -> None:
        # No mutable state to reset.
        return

    def forward(
        self,
        generated_graphs: List[Any],
        ref_metrics: Optional[Dict[str, Any]] = None,
        local_rank: int = 0,
        test: bool = False,
        sampled_generated: Optional[List[nx.Graph]] = None,
    ) -> Dict[str, float]:
        reference_graphs = self.test_graphs if test else self.val_graphs
        split = "test" if test else "validation"

        if local_rank == 0:
            print(
                f"Computing wireless metrics between {len(generated_graphs)} generated graphs "
                f"and {len(reference_graphs)} {split} graphs."
            )

        edge_metrics = self._edge_distribution_metrics(reference_graphs, generated_graphs)

        if sampled_generated is None:
            sampled_generated = self._sample_subgraphs(generated_graphs, len(reference_graphs))

        weighted_degree = self._weighted_degree_stats(reference_graphs, sampled_generated)

        spectre = spectral_stats(
            reference_graphs,
            sampled_generated,
            is_parallel=True,
            n_eigvals=-1,
            compute_emd=False,
        )

        metrics_out = {
            "edge_ks": edge_metrics["edge_ks_mean"],
            "edge_wasserstein": edge_metrics["edge_wasserstein_mean"],
            "edge_pairs_used": edge_metrics["edge_pairs_used"],
            "degree_weighted": weighted_degree,
            "spectre": spectre,
        }

        if wandb.run:
            for key, val in metrics_out.items():
                wandb.run.summary[f"{split}/{key}"] = val

        if local_rank == 0 and ref_metrics:
            ref_split = ref_metrics.get("test" if test else "val") or {}
            if ref_split:
                print(f"Reference metrics available for {split}: {list(ref_split.keys())}")

        return metrics_out

    @staticmethod
    def _loader_to_nx(loader):
        networkx_graphs = []
        for batch in loader:
            adjs = batch[1]
            G = adjs_to_graphs(adjs, is_cuda=True)
            networkx_graphs.extend(G)
        return networkx_graphs

    def _collect_edge_values(self, graphs: List[nx.Graph]) -> Dict[tuple, List[float]]:
        edge_vals: Dict[tuple, List[float]] = {}
        for g in graphs:
            for u, v, data in g.edges(data=True):
                # Use strict physical parameters for statistical edge comparisons across the sets 
                val = float(data.get("interference_raw", data.get("weight", 0.0)))
                key = tuple(sorted((int(u), int(v))))
                edge_vals.setdefault(key, []).append(val)
        return edge_vals

    def _edge_distribution_metrics(self, reference_graphs: List[nx.Graph], generated_graphs: List[nx.Graph]) -> Dict[str, float]:
        ref_vals = self._collect_edge_values(reference_graphs)
        gen_vals = self._collect_edge_values(generated_graphs)

        ks_scores = []
        wass_scores = []
        for key in set(ref_vals.keys()).union(gen_vals.keys()):
            ref = ref_vals.get(key, [])
            gen = gen_vals.get(key, [])
            if len(ref) == 0 or len(gen) == 0:
                continue
            ks_scores.append(ks_2samp(ref, gen).statistic)
            wass_scores.append(wasserstein_distance(ref, gen))

        edge_pairs_used = len(ks_scores)
        edge_ks_mean = float(np.mean(ks_scores)) if ks_scores else 0.0
        edge_wass_mean = float(np.mean(wass_scores)) if wass_scores else 0.0

        return {
            "edge_pairs_used": edge_pairs_used,
            "edge_ks_mean": edge_ks_mean,
            "edge_wasserstein_mean": edge_wass_mean,
        }

    @staticmethod
    def _graph_size_distribution(graphs: List[nx.Graph]) -> tuple:
        sizes = np.array([max(1, g.number_of_nodes()) for g in graphs], dtype=int)
        support, counts = np.unique(sizes, return_counts=True)
        probs = counts / counts.sum() if counts.sum() > 0 else np.array([1.0])
        return support, probs

    def _sample_subgraphs(self, graphs: List[nx.Graph], num_samples: int) -> List[nx.Graph]:
        if len(self.test_size_support) == 0 or len(graphs) == 0:
            return []
        sampled_graphs: List[nx.Graph] = []
        sizes = self.rng.choice(self.test_size_support, size=num_samples, p=self.test_size_probs)
        for target_size in sizes:
            idx = self.rng.choice(len(graphs))
            g = graphs[idx]
            nodes = list(g.nodes())
            if len(nodes) <= target_size:
                sampled_graphs.append(g.copy())
                continue
            chosen = self.rng.choice(nodes, size=int(target_size), replace=False)
            sampled_graphs.append(g.subgraph(chosen).copy())
        return sampled_graphs

    @staticmethod
    def _weighted_degree_stats_worker(G, bins):
        degrees = []
        for n in G.nodes():
            # Use `weight_norm` `[0,1]` scale for generated if available, otherwise `weight` `[0,1]` for test set. 
            # Summing negative dBm values physically misaligns degree comparisons.
            deg = sum(data.get("weight_norm", data.get("weight", 0.0)) for _, _, data in G.edges(n, data=True))
            degrees.append(float(deg))
        hist, _ = np.histogram(degrees, bins=bins, density=False)
        return hist

    def _weighted_degree_stats(self, reference_graphs: List[nx.Graph], generated_graphs: List[nx.Graph]) -> float:
        all_degrees = []
        for g in reference_graphs + generated_graphs:
            for n in g.nodes():
                deg = sum(data.get("weight_norm", data.get("weight", 0.0)) for _, _, data in g.edges(n, data=True))
                all_degrees.append(float(deg))

        max_deg = max(all_degrees) if all_degrees else 1.0
        min_deg = min(all_degrees) if all_degrees else 0.0
        if max_deg == min_deg:
            max_deg = min_deg + 1.0
        bins = np.linspace(min_deg, max_deg, num=51)

        ref_hists = []
        pred_hists = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for hist in executor.map(self._weighted_degree_stats_worker, reference_graphs, [bins] * len(reference_graphs)):
                ref_hists.append(hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for hist in executor.map(self._weighted_degree_stats_worker, generated_graphs, [bins] * len(generated_graphs)):
                pred_hists.append(hist)

        return compute_mmd(ref_hists, pred_hists, kernel=gaussian_tv, is_hist=True)
