from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils import graph_list_to_dataset


class WirelessDatasetModule(pl.LightningDataModule):
    """
    Lightning datamodule to serve MetroFi wireless interference graphs.

    The underlying pickle is expected to contain the keys 'train', 'val', 'test',
    and an optional 'metadata' entry with global MAC / location information.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.data.batch_size
        self.max_node_num = config.data.max_node_num
        self.max_feat_num = config.data.max_feat_num
        self.init_type = config.data.init
        self.min_observed_nodes = getattr(config.data, "min_observed_nodes", 3)
        self.max_train_graphs = getattr(config.data, "max_train_graphs", None)
        self.max_val_graphs = getattr(config.data, "max_val_graphs", None)
        self.max_test_graphs = getattr(config.data, "max_test_graphs", None)
        dataset_dir = Path(config.data.dir)
        dataset_name = f"{config.data.data}.pkl"
        self.data_path = dataset_dir / dataset_name
        self.val_split = getattr(config.data, "val_split", 0.1)
        self.test_split = getattr(config.data, "test_split", 0.1)

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.metadata: Dict[str, Any] = {}
        self.conditional = bool(getattr(config.model, "conditional", False))
        self.coord_min = None
        self.coord_scale = None

    def setup(self, stage: Optional[str] = None):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Expected dataset at {self.data_path}. Run tools/build_metrofi_dataset.py first.")

        dataset = pd.read_pickle(self.data_path)
        self.metadata = dataset.get("metadata", {})

        self._normalise_graphs(dataset)

        self.train_graphs = self._limit_graphs(
            self._filter_graphs(dataset.get("train", [])), self.max_train_graphs, "train"
        )
        self.val_graphs = self._limit_graphs(
            self._filter_graphs(dataset.get("val", [])), self.max_val_graphs, "val"
        )
        self.test_graphs = self._limit_graphs(
            self._filter_graphs(dataset.get("test", [])), self.max_test_graphs, "test"
        )

        all_graphs = self.train_graphs + self.val_graphs + self.test_graphs
        self.interference_range = self._interference_range(all_graphs)
        if self.conditional:
            self._setup_coord_normalizer(all_graphs)
        if self.interference_range is not None:
            vmin, vmax = self.interference_range
            print(f"WirelessDatasetModule: interference range (post-filter) min={vmin:.4g}, max={vmax:.4g}")
        else:
            print("WirelessDatasetModule: no edges found to compute interference range after filtering.")

        self.train_ds = self._build_dataset(self.train_graphs)
        self.val_ds = self._build_dataset(self.val_graphs)
        self.test_ds = self._build_dataset(self.test_graphs)


    @staticmethod
    def _limit_graphs(graphs, max_graphs, split_name):
        if max_graphs is None:
            return graphs
        limited = graphs[: int(max_graphs)]
        print(f"WirelessDatasetModule: using {len(limited)} {split_name} graphs after debug limit.")
        return limited

    def _build_dataset(self, graphs):
        dataset = graph_list_to_dataset(
            graphs,
            self.init_type,
            self.max_node_num,
            self.max_feat_num,
            mask_attr="observed",
        )
        if not self.conditional:
            return dataset

        coords = self._graph_coords_tensor(graphs)
        return TensorDataset(*dataset.tensors, coords)

    def _setup_coord_normalizer(self, graphs):
        coords = []
        for graph in graphs:
            coord = graph.graph.get("coordinates")
            if coord is not None:
                coords.append([float(coord[0]), float(coord[1])])

        if not coords:
            self.coord_min = torch.zeros(2, dtype=torch.float32)
            self.coord_scale = torch.ones(2, dtype=torch.float32)
            return

        coords_t = torch.tensor(coords, dtype=torch.float32)
        self.coord_min = coords_t.min(dim=0).values
        coord_max = coords_t.max(dim=0).values
        self.coord_scale = (coord_max - self.coord_min).clamp_min(1e-12)

    def _graph_coords_tensor(self, graphs):
        if self.coord_min is None or self.coord_scale is None:
            self._setup_coord_normalizer(graphs)

        coords = []
        for graph in graphs:
            coord = graph.graph.get("coordinates")
            if coord is None:
                coords.append(torch.zeros(2, dtype=torch.float32))
                continue
            coord_t = torch.tensor([float(coord[0]), float(coord[1])], dtype=torch.float32)
            coords.append((coord_t - self.coord_min) / self.coord_scale)

        if not coords:
            return torch.zeros((0, 2), dtype=torch.float32)
        return torch.stack(coords, dim=0)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

    def node_counts(self, max_nodes_possible: int = 1000):
        if self.train_ds is None or self.val_ds is None:
            raise RuntimeError("Call setup() before requesting node counts.")

        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for batch in loader:
                adjs = batch[1]
                observed_mask = batch[2] if len(batch) > 2 else None
                for idx, A in enumerate(adjs):
                    if observed_mask is not None:
                        num_nodes = int(observed_mask[idx].sum().item())
                    else:
                        num_nodes = int((A.sum(dim=1) != 0).sum().item())
                    if num_nodes < max_nodes_possible:
                        all_counts[num_nodes] += 1
        nz = all_counts.nonzero(as_tuple=False)
        if nz.numel() == 0:
            return all_counts
        max_index = int(nz.max())
        all_counts = all_counts[: max_index + 1]
        total = all_counts.sum()
        if total > 0:
            all_counts = all_counts / total
        return all_counts

    def num_mac_addresses(self) -> int:
        macs = self.metadata.get("mac_id_to_address") or []
        return len(macs)

    def mac_addresses(self):
        return list(self.metadata.get("mac_id_to_address", []))

    def location_coordinates(self):
        return dict(self.metadata.get("location_id_to_coords", {}))

    def _filter_graphs(self, graphs):
        filtered = []
        dropped = 0
        for graph in graphs:
            observed = [
                node
                for node, data in graph.nodes(data=True)
                if data.get("observed")
                or data.get("sample_count", 0) > 0
                or (data.get("mean_rssi") is not None and not pd.isna(data.get("mean_rssi")))
            ]
            if len(observed) >= self.min_observed_nodes:
                filtered.append(graph)
            else:
                dropped += 1
        if dropped > 0:
            print(
                f"WirelessDatasetModule: filtered out {dropped} graphs with fewer than "
                f"{self.min_observed_nodes} observed nodes."
            )
        return filtered

    @staticmethod
    def _interference_range(graphs):
        values = []
        for g in graphs:
            for _, _, data in g.edges(data=True):
                val = data.get("interference_raw", data.get("interference", data.get("weight")))
                if val is not None:
                    values.append(float(val))
        if not values:
            return None
        return min(values), max(values)

    def _normalise_graphs(self, dataset):
        # Compute global min/max across all splits before normalising.
        min_raw = None
        max_raw = None
        for split in ("train", "val", "test"):
            for graph in dataset.get(split, []):
                for _, _, data in graph.edges(data=True):
                    raw = float(data.get("interference", data.get("weight", 0.0)))
                    min_raw = raw if min_raw is None else min(min_raw, raw)
                    max_raw = raw if max_raw is None else max(max_raw, raw)

        if min_raw is None or max_raw is None:
            # No edges; nothing to normalise.
            self.interference_min = 0.0
            self.interference_max = 0.0
            self.max_interference = 0.0
            return

        scale = max_raw - min_raw
        if scale <= 0:
            scale = 1.0

        for split in ("train", "val", "test"):
            for graph in dataset.get(split, []):
                for _, _, data in graph.edges(data=True):
                    raw = float(data.get("interference", data.get("weight", 0.0)))
                    norm = (raw - min_raw) / scale if scale > 0 else 0.0
                    data["interference_raw"] = raw
                    data["interference_clipped"] = raw  # retain key for downstream use
                    data["weight"] = norm
            graph.graph["interference_scale"] = scale
            graph.graph["interference_min"] = min_raw
            graph.graph["interference_max"] = max_raw

        self.interference_min = float(min_raw)
        self.interference_max = float(max_raw)
