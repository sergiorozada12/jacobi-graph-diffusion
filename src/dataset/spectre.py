import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import networkx as nx

from src.utils import graph_list_to_dataset


class SpectreDatasetModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.data.batch_size
        self.max_node_num = config.data.max_node_num
        self.max_feat_num = config.data.max_feat_num
        self.init_type = config.data.init
        data_name = config.data.data
        if isinstance(data_name, str) and os.path.isabs(data_name):
            self.data_path = data_name
        else:
            rel_path = data_name if data_name.endswith(".pkl") else data_name + ".pkl"
            self.data_path = os.path.join(config.data.dir, rel_path)
        self.test_split = config.data.test_split
        self.val_split = config.data.val_split

    def setup(self, stage=None):
        with open(self.data_path, "rb") as f:
            dataset = pickle.load(f)

        self.train_graphs = dataset["train"]
        self.val_graphs = dataset["val"]
        self.test_graphs = dataset["test"]

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
                adjs = batch[1]
                for A in adjs:
                    num_nodes = (A.sum(dim=1) != 0).sum().item()
                    all_counts[num_nodes] += 1
                    
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[: max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts
