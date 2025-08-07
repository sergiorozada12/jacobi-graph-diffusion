import torch
import numpy as np
import networkx as nx
from torch.utils.data import TensorDataset


def node_flags(adjs, eps=1e-5):
    """
    Compute binary flags for active nodes based on adjacency matrix.
    
    Args:
        adjs (Tensor): Shape [B, N, N] or [B, C, N, N].

    Returns:
        Tensor: Binary flags of shape [B, N].
    """
    flags = (adjs.abs().sum(-1) > eps).float()
    if flags.ndim == 3:
        flags = flags[:, 0, :]
    return flags.to(adjs.device)


def mask_adjs(adjs, flags):
    """
    Apply masking to adjacency matrices using node flags.
    
    Args:
        adjs (Tensor): Shape [B, N, N] or [B, C, N, N].
        flags (Tensor): Shape [B, N].

    Returns:
        Tensor: Masked adjacency matrices.
    """
    if flags is None:
        flags = torch.ones((adjs.shape[0], adjs.shape[-1]), device=adjs.device)

    if adjs.dim() == 4:
        flags = flags.unsqueeze(1)  # [B, 1, N]

    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs


def mask_x(x, flags):
    """
    Mask node features using binary node flags.
    
    Args:
        x (Tensor): Node features [B, N, F].
        flags (Tensor): Node flags [B, N].

    Returns:
        Tensor: Masked node features.
    """
    return x * flags.unsqueeze(-1)


def gen_noise(adj, flags):
    """
    Generate symmetric Gaussian noise respecting graph structure.
    
    Args:
        adj (Tensor): Adjacency tensor [B, N, N].
        flags (Tensor): Node flags [B, N].

    Returns:
        Tensor: Symmetric noise tensor.
    """
    z = torch.randn_like(adj).triu(1)
    z = z + z.transpose(-1, -2)
    return mask_adjs(z, flags)


def quantize(adjs, thr=0.5):
    """
    Threshold adjacency matrices to obtain binary edges.

    Args:
        adjs (Tensor): Adjacency matrices [B, N, N].
        thr (float): Threshold.

    Returns:
        Tensor: Binarized adjacency matrices.
    """
    return torch.where(adjs < thr, torch.zeros_like(adjs), torch.ones_like(adjs))


def adjs_to_graphs(adjs, is_cuda=False):
    """
    Convert adjacency matrices to NetworkX graphs.

    Args:
        adjs (Tensor): Shape [B, N, N].
        is_cuda (bool): Whether tensors are on CUDA.

    Returns:
        List[nx.Graph]: List of NetworkX graphs.
    """
    graph_list = []
    for adj in adjs:
        if is_cuda:
            adj = adj.detach().cpu().numpy()
        G = nx.from_numpy_array(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 1:
            G.add_node(1)
        graph_list.append(G)
    return graph_list


def graphs_to_tensor(graph_list, max_node_num):
    """
    Convert list of graphs to padded adjacency tensor.

    Args:
        graph_list (List[nx.Graph]): List of graphs.
        max_node_num (int): Max node limit per graph.

    Returns:
        Tensor: Tensor of shape [B, N, N].
    """
    adjs_list = []
    for g in graph_list:
        node_list = list(g.nodes())
        adj = nx.to_numpy_array(g, nodelist=node_list)
        padded = pad_adjs(adj, max_node_num)
        adjs_list.append(padded)

    adjs_np = np.stack(adjs_list)
    return torch.tensor(adjs_np, dtype=torch.float32)


def pad_adjs(adj, max_node_num):
    """
    Pad 2D adjacency matrix to max_node_num x max_node_num.

    Args:
        adj (ndarray): Shape [n, n].
        max_node_num (int): Target dimension.

    Returns:
        ndarray: Padded matrix.
    """
    n = adj.shape[0]
    if n > max_node_num:
        raise ValueError(f"Graph with {n} nodes exceeds max_node_num={max_node_num}")
    padded = np.zeros((max_node_num, max_node_num), dtype=np.float32)
    padded[:n, :n] = adj
    return padded


def init_features(adjs, init_type, max_feat_num):
    """
    Initialize node features, then mask them.

    Args:
        adjs (Tensor): Adjacency tensor [B, N, N].
        init_type (str): "zeros" or "ones".
        max_feat_num (int): Number of features per node.

    Returns:
        Tensor: Node feature tensor [B, N, F].
    """
    if init_type == "zeros":
        features = torch.zeros(adjs.size(0), adjs.size(1), max_feat_num, dtype=torch.float32)
    elif init_type == "ones":
        features = torch.ones(adjs.size(0), adjs.size(1), max_feat_num, dtype=torch.float32)
    else:
        raise NotImplementedError(f"Unknown init type {init_type}")
    flags = node_flags(adjs)
    return mask_x(features, flags)


def graph_list_to_dataset(graph_list, init_type, max_node_num, max_feat_num):
    """
    Convert list of graphs to a PyTorch dataset.

    Args:
        graph_list (List[nx.Graph]): List of graphs.
        init_type (str): Feature init mode ("zeros" or "ones").
        max_node_num (int): Max number of nodes per graph.
        max_feat_num (int): Number of node features.

    Returns:
        TensorDataset: Dataset with (features, adjs).
    """
    adjs = graphs_to_tensor(graph_list, max_node_num)
    features = init_features(adjs, init_type, max_feat_num)
    return TensorDataset(features, adjs)


def init_flags(graph_list, config, batch_size=None):
    """
    Generate node flags for a batch from a graph list.

    Args:
        graph_list (List[nx.Graph]): List of graphs.
        config (OmegaConf): Full configuration.
        batch_size (int, optional): Sample size.

    Returns:
        Tensor: Node flags [B, N].
    """
    if batch_size is None:
        batch_size = config.data.batch_size
    max_node_num = config.data.max_node_num
    graph_tensor = graphs_to_tensor(graph_list, max_node_num)
    idx = np.random.randint(0, len(graph_list), batch_size)
    flags = node_flags(graph_tensor[idx])
    return flags


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def to_device(self, device):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.to(device)
        self.E = self.E.to(device)
        self.y = self.y.to(device) if self.y is not None else None
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

    def __repr__(self):
        return (
            f"X: {self.X.shape if type(self.X) == torch.Tensor else self.X} -- "
            + f"E: {self.E.shape if type(self.E) == torch.Tensor else self.E} -- "
            + f"y: {self.y.shape if type(self.y) == torch.Tensor else self.y}"
        )

    def split(self, node_mask):
        """Split a PlaceHolder representing a batch into a list of placeholders representing individual graphs."""
        graph_list = []
        batch_size = self.X.shape[0]
        for i in range(batch_size):
            n = torch.sum(node_mask[i], dim=0)
            x = self.X[i, :n]
            e = self.E[i, :n, :n]
            y = self.y[i] if self.y is not None else None
            graph_list.append(PlaceHolder(X=x, E=e, y=y))
        return graph_list
