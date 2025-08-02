import os
import pickle
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.utils import graph_list_to_dataset


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
            graph_list = pickle.load(f)

        np.random.seed(self.config.general.seed)
        np.random.shuffle(graph_list)

        n_total = len(graph_list)
        n_test = int(n_total * self.test_split)
        n_val = int(n_total * self.val_split)
        n_train = n_total - n_test - n_val

        self.train_graphs = graph_list[:n_train]
        self.val_graphs = graph_list[n_train:n_train + n_val]
        self.test_graphs = graph_list[n_train + n_val:]

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
