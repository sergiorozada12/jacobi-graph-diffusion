import copy
import os
from pathlib import Path
import torch

from src.utils import adjs_to_graphs


class DistributionNodes:
    def __init__(self, prob):
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)


def resolve_size_ref_dataset_path(data_dir, dataset_name, target_nodes):
    return Path(data_dir) / "size_ref" / dataset_name / f"{dataset_name}_{target_nodes}.pkl"


def compute_reference_metrics(datamodule, sampling_metrics, cache_name=None):
    dataset_name = cache_name or getattr(datamodule.config.data, "data", "dataset")
    metrics_dir = os.path.join(datamodule.config.data.dir, "ref_metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"ref_metrics_{dataset_name}.pt")

    if os.path.exists(metrics_path):
        print(f"Loading cached sampling metrics from {metrics_path}.")
        ref_metrics = torch.load(metrics_path, map_location="cpu")
        print(f"Reference metrics loaded: keys={list(ref_metrics.keys())}")
        for split, metrics in ref_metrics.items():
            if metrics is None:
                continue
            print(f"  {split}: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}" for k, v in metrics.items()))
        return ref_metrics

    print("Computing sampling metrics.")
    training_graphs = []
    print("Converting training dataset to format required by sampling metrics.")
    for data_batch in datamodule.train_dataloader():
        A = data_batch[1]
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

    ref_metrics = {
        'val': val_ref_metrics,
        'test': test_ref_metrics
    }

    print("Computed reference metrics:")
    for split, metrics in ref_metrics.items():
        if metrics is None:
            continue
        print(f"  {split}: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}" for k, v in metrics.items()))

    torch.save(ref_metrics, metrics_path)
    print(f"Saved sampling metrics to {metrics_path}.")
    return ref_metrics


def load_size_ref_metrics(cfg, metrics_cls, target_nodes):
    from src.dataset.spectre import SpectreDatasetModule

    size_ref_path = resolve_size_ref_dataset_path(cfg.data.dir, cfg.data.data, target_nodes)
    if not size_ref_path.exists():
        raise FileNotFoundError(
            f"Size-ref dataset not found for dataset '{cfg.data.data}' and target_nodes={target_nodes}: {size_ref_path}"
        )

    ref_cfg = copy.deepcopy(cfg)
    ref_cfg.data.data = str(size_ref_path)
    ref_datamodule = SpectreDatasetModule(ref_cfg)
    ref_datamodule.setup()
    ref_sampling_metrics = metrics_cls(ref_datamodule)
    cache_name = f"{cfg.data.data}_size_{target_nodes}"
    size_ref_metrics = compute_reference_metrics(
        ref_datamodule,
        ref_sampling_metrics,
        cache_name=cache_name,
    )
    return size_ref_metrics
