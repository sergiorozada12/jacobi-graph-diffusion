import copy
import torch

from src.utils import adjs_to_graphs


class DistributionNodes:
    def __init__(self, prob):
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)


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