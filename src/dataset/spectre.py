import os
import copy
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import networkx as nx

from src.utils import graph_list_to_dataset, quantize, adjs_to_graphs


class SynthGraphDatasetModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.data.batch_size
        self.max_node_num = config.data.max_node_num
        self.max_feat_num = config.data.max_feat_num
        self.init_type = config.data.init
        self.data_path = os.path.join(config.data.dir, config.data.data + ".pkl")
        self.test_split = config.data.test_split
        self.val_split = config.data.val_split

    def setup(self, stage=None):
        with open(self.data_path, "rb") as f:
            dataset = pickle.load(f)

        train_graphs = dataset["train"]
        val_graphs = dataset["val"]
        test_graphs = dataset["test"]

        self.train_graphs = [
            nx.to_numpy_array(graph).fill_diagonal_(0)
            for graph in train_graphs
        ]
        self.val_graphs = [
            nx.to_numpy_array(graph).fill_diagonal_(0)
            for graph in val_graphs
        ]
        self.test_graphs = [
            nx.to_numpy_array(graph).fill_diagonal_(0)
            for graph in test_graphs
        ]

        self.train_ds = graph_list_to_dataset(
            self.train_graphs,
            self.init_type,
            self.max_node_num,
            self.max_feat_num
        )
        self.val_ds = graph_list_to_dataset(
            self.val_graphs,
            self.init_type,
            self.max_node_num,
            self.max_feat_num
        )
        self.test_ds = graph_list_to_dataset(
            self.test_graphs,
            self.init_type,
            self.max_node_num,
            self.max_feat_num
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

    def node_counts(self, max_nodes_possible=1000):
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for batch in loader:
                _, adjs = batch
                for A in adjs:
                    num_nodes = (A.sum(dim=1) != 0).sum().item()
                    all_counts[num_nodes] += 1
                    
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[: max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

def compute_reference_metrics(datamodule, sampling_metrics):
    print("Computing sampling metrics.")
    training_graphs = []
    print("Converting training dataset to format required by sampling metrics.")
    for data_batch in datamodule.train_dataloader():
        _, A = data_batch
        G = adjs_to_graphs(A, is_cuda=True)
        training_graphs.extend(G)

    dummy_kwargs = {
        "local_rank": 0,
        "ref_metrics": {"val": None, "test": None},
    }

    print("Computing validation reference metrics.")
    val_sampling_metrics = copy.deepcopy(sampling_metrics)

    val_ref_metrics = val_sampling_metrics.forward(
        training_graphs,
        test=False,
        **dummy_kwargs,
    )

    print("Computing test reference metrics.")
    test_sampling_metrics = copy.deepcopy(sampling_metrics)
    test_ref_metrics = test_sampling_metrics.forward(
        training_graphs,
        test=True,
        **dummy_kwargs,
    )

    return {
        'val': val_ref_metrics,
        'test': test_ref_metrics
    }
