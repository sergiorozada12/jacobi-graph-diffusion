import math
import torch
import numpy as np
import networkx as nx
from torch.utils.data import TensorDataset


def build_time_schedule(num_scales, T, eps, kind="log", power=2.0):
    """Return a decreasing timestep sequence between ``T`` and ``eps``."""

    if num_scales <= 0:
        raise ValueError("Number of discretization steps must be positive")

    schedule = (kind or "log").lower()
    T = float(T)
    eps = float(eps)

    if eps <= 0.0 or T <= 0.0:
        raise ValueError("Both T and eps must be positive")

    steps = torch.linspace(0.0, 1.0, num_scales + 1, dtype=torch.float64)

    if schedule == "linear":
        ts = torch.linspace(T, eps, num_scales + 1, dtype=torch.float64)
    elif schedule == "power":
        weights = steps.pow(power)
        ts = T - (T - eps) * weights
    elif schedule == "cosine":
        weights = 0.5 * (1.0 - torch.cos(math.pi * steps))
        ts = T - (T - eps) * weights
    elif schedule == "log":
        ts = torch.exp(torch.linspace(math.log(T), math.log(eps), num_scales + 1, dtype=torch.float64))
    elif schedule == "log_power":
        weights = steps.pow(power)
        log_T = math.log(T)
        log_eps = math.log(eps)
        ts = torch.exp(log_T + weights * (log_eps - log_T))
    elif schedule == "double_log":
        a = power if power is not None else 3.0
        weights = 1.0 - torch.exp(-a * steps)
        log_T = math.log(T)
        log_eps = math.log(eps)
        ts = torch.exp(log_eps + weights * (log_T - log_eps))
    else:
        raise ValueError(f"Unknown time schedule '{kind}'")

    ts[0] = T
    ts[-1] = eps
    ts = ts.clamp(min=eps, max=T)
    return ts.to(dtype=torch.float32)


def node_flags(adjs, observed_mask=None, eps=1e-5):
    """
    Compute binary flags for active nodes.
    
    Args:
        adjs (Tensor): Shape [B, N, N] or [B, C, N, N].
        observed_mask (Tensor, optional): Pre-computed mask indicating which nodes are observed.
        eps (float): Threshold for detecting active nodes when observed_mask is None.

    Returns:
        Tensor: Binary flags of shape [B, N].
    """
    if observed_mask is not None:
        return observed_mask.to(adjs.device).float()

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
        return adjs

    # Support [B, N, N] and [B, N, N, K]
    B, N = flags.shape
    if adjs.dim() == 3:
        # [B, N, 1] * [B, 1, N]
        adjs = adjs * flags.unsqueeze(-1) * flags.unsqueeze(-2)
    elif adjs.dim() == 4:
        # [B, N, 1, 1] * [B, 1, N, 1]
        adjs = adjs * flags.view(B, N, 1, 1) * flags.view(B, 1, N, 1)
    
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


def mask_and_clamp(state, flags):
    """
    Unified masking and clamping for both Tensors and PlaceHolders.
    Handles symmetry for adjacency matrices and categorical dimensions.
    """
    is_placeholder = hasattr(state, "X") and hasattr(state, "E")
    
    if not is_placeholder:
        # Legacy/Scalar mode
        state = state.clamp(0.0, 1.0)
        state = mask_adjs(state, flags)
        # Symmetrize scalar adjacency
        if state.dim() == 3:
            state_triu = torch.triu(state, diagonal=1)
            state = state_triu + state_triu.transpose(-1, -2)
        return state
    else:
        # Joint state (PlaceHolder)
        E = state.E.clamp(0.0, 1.0)
        E = mask_adjs(E, flags)
        # Symmetrize E
        if E.ndim == 3:
            E_triu = torch.triu(E, diagonal=1)
            E = E_triu + E_triu.transpose(-1, -2)
        else:
            # Simplex E: [B, N, N, K]
            # Categorical E symmetry: transpose indices N, N but keep K
            E = 0.5 * (E + E.transpose(1, 2))
            
        X = state.X
        if X is not None:
            X = X.clamp(0.0, 1.0)
            X = mask_x(X, flags)
            
        from src.utils import PlaceHolder
        return PlaceHolder(X=X, E=E, y=state.y)


def gen_noise(state, flags):
    """
    Generate symmetric Gaussian noise respecting graph structure.
    Supports both Tensors and PlaceHolders.
    """
    is_placeholder = hasattr(state, "X") and hasattr(state, "E")
    
    if not is_placeholder:
        # Legacy/Scalar mode
        z = torch.randn_like(state)
        # Symmetrize if 3D square (adjacency)
        if z.ndim == 3 and z.shape[1] == z.shape[2]:
            z = torch.triu(z, diagonal=1)
            z = z + z.transpose(-1, -2)
            return mask_adjs(z, flags)
        # Symmetrize if 4D categorical edge [B, N, N, K]
        elif z.ndim == 4 and z.shape[1] == z.shape[2]:
            z = 0.5 * (z + z.transpose(1, 2))
            return mask_adjs(z, flags)
        else:
            return mask_x(z, flags)
    else:
        # Joint state (PlaceHolder)
        # Noise for E (Adjacency or Bond types)
        ze = torch.randn_like(state.E)
        if ze.ndim == 3:
            ze = torch.triu(ze, diagonal=1)
            ze = ze + ze.transpose(-1, -2)
        elif ze.ndim == 4:
            # Simplex E: [B, N, N, K]
            ze = 0.5 * (ze + ze.transpose(1, 2))
        ze = mask_adjs(ze, flags)
        
        # Noise for X (Node features)
        zx = None
        if state.X is not None:
            zx = torch.randn_like(state.X)
            zx = mask_x(zx, flags)
            
        from src.utils import PlaceHolder
        return PlaceHolder(X=zx, E=ze, y=state.y)


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


def adjs_to_graphs(adjs, is_cuda=False, keep_isolates=False, nodelist=None, keep_zero_weights=False):
    """
    Convert adjacency matrices to NetworkX graphs.

    Args:
        adjs (Tensor): Shape [B, N, N].
        is_cuda (bool): Whether tensors are on CUDA.
        keep_isolates (bool): Whether to retain isolated nodes.
        nodelist (Sequence, optional): Node labels to use. Must have length N if provided.
        keep_zero_weights (bool): Whether to include zero-weight edges.

    Returns:
        List[nx.Graph]: List of NetworkX graphs.
    """
    graph_list = []
    for adj in adjs:
        if isinstance(adj, torch.Tensor):
            adj_np = adj.detach().cpu().numpy()
        else:
            adj_np = np.asarray(adj)
        num_nodes = adj_np.shape[0]
        nodes = list(range(num_nodes)) if nodelist is None else list(nodelist)
        if len(nodes) != num_nodes:
            raise ValueError(f"nodelist length {len(nodes)} does not match adjacency size {num_nodes}.")

        G = nx.Graph()
        G.add_nodes_from(nodes)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                w = float(adj_np[i, j])
                if keep_zero_weights or w != 0.0:
                    G.add_edge(nodes[i], nodes[j], weight=w)
        if not keep_isolates:
            G.remove_nodes_from(list(nx.isolates(G)))
            if G.number_of_nodes() < 1:
                G.add_node(nodes[0] if nodes else 0)
        graph_list.append(G)
    return graph_list


def graphs_to_tensor(graph_list, max_node_num, mask_attr=None):
    """
    Convert list of graphs to padded adjacency tensor.

    Args:
        graph_list (List[nx.Graph]): List of graphs.
        max_node_num (int): Max node limit per graph.
        mask_attr (str, optional): Node attribute storing an observed flag. If provided,
            a tensor of shape [B, N] with the observed mask is also returned.

    Returns:
        Tensor or Tuple[Tensor, Tensor]: Adjacency tensor [B, N, N] and optionally
        an observed mask tensor [B, N].
    """
    adjs_list = []
    mask_list = [] if mask_attr is not None else None
    for g in graph_list:
        node_list = list(g.nodes())
        adj = nx.to_numpy_array(g, nodelist=node_list)
        np.fill_diagonal(adj, 0)
        padded = pad_adjs(adj, max_node_num)
        adjs_list.append(padded)

        if mask_attr is not None:
            mask = np.zeros(max_node_num, dtype=np.float32)
            num_nodes = min(len(node_list), max_node_num)
            for idx in range(num_nodes):
                node = node_list[idx]
                mask[idx] = float(bool(g.nodes[node].get(mask_attr, False)))
            mask_list.append(mask)

    adjs_np = np.stack(adjs_list)
    adjs_tensor = torch.tensor(adjs_np, dtype=torch.float32)

    if mask_attr is not None:
        masks_np = np.stack(mask_list)
        mask_tensor = torch.tensor(masks_np, dtype=torch.float32)
        return adjs_tensor, mask_tensor
    return adjs_tensor


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


def init_features(adjs, init_type, max_feat_num, observed_mask=None):
    """
    Initialize node features, then mask them.

    Args:
        adjs (Tensor): Adjacency tensor [B, N, N].
        init_type (str): "zeros" or "ones".
        max_feat_num (int): Number of features per node.
        observed_mask (Tensor, optional): Explicit node mask to apply.

    Returns:
        Tensor: Node feature tensor [B, N, F].
    """
    if init_type == "zeros":
        features = torch.zeros(adjs.size(0), adjs.size(1), max_feat_num, dtype=torch.float32)
    elif init_type == "ones":
        features = torch.ones(adjs.size(0), adjs.size(1), max_feat_num, dtype=torch.float32)
    else:
        raise NotImplementedError(f"Unknown init type {init_type}")
    flags = node_flags(adjs, observed_mask)
    return mask_x(features, flags)


def graph_list_to_dataset(
    graph_list,
    init_type,
    max_node_num,
    max_feat_num,
    mask_attr=None,
):
    """
    Convert list of graphs to a PyTorch dataset.

    Args:
        graph_list (List[nx.Graph]): List of graphs.
        init_type (str): Feature init mode ("zeros" or "ones").
        max_node_num (int): Max number of nodes per graph.
        max_feat_num (int): Number of node features.
        mask_attr (str, optional): Node attribute indicating observed nodes to
            include in the returned dataset.

    Returns:
        TensorDataset: Dataset with (features, adjs).
    """
    tensor_out = graphs_to_tensor(graph_list, max_node_num, mask_attr=mask_attr)
    if mask_attr is not None:
        adjs, masks = tensor_out
    else:
        adjs = tensor_out
        masks = None
    features = init_features(adjs, init_type, max_feat_num, observed_mask=masks)
    if masks is not None:
        return TensorDataset(features, adjs, masks)
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

    def to(self, *args, **kwargs):
        if self.X is not None:
            self.X = self.X.to(*args, **kwargs)
        if self.E is not None:
            self.E = self.E.to(*args, **kwargs)
        if self.y is not None:
            self.y = self.y.to(*args, **kwargs)
        return self

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        if self.X is not None:
            self.X = self.X.type_as(x)
        if self.E is not None:
            self.E = self.E.type_as(x)
        if self.y is not None:
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

def assert_symmetric_and_masked(adj: torch.Tensor, flags: torch.Tensor, tol: float = 1e-6):
    """
    adj:   [B, N, N]
    flags: [B, N] (1 for active nodes, 0 for inactive)
    tol:   tolerance for numerical checks
    """
    B, N, _ = adj.shape

    # Symmetry check
    diff_sym = (adj - adj.transpose(-1, -2)).abs().max().item()
    assert diff_sym < tol, f"Adj not symmetric, max diff = {diff_sym}"

    # Diagonal zero check
    diag_vals = adj.diagonal(dim1=-2, dim2=-1)
    max_diag = diag_vals.abs().max().item()
    assert max_diag < tol, f"Diagonal not zero, max diag = {max_diag}"

    # Flags masking check (inactive rows/cols must be all zero)
    flags_mask = (flags[:, :, None] * flags[:, None, :]).float()  # [B,N,N]
    masked = adj * (1 - flags_mask)
    max_off = masked.abs().max().item()
    assert max_off < tol, f"Non-flagged entries not zero, max = {max_off}"


def assert_symmetric_and_masked_E(E: torch.Tensor, flags: torch.Tensor, tol: float = 1e-6):
    """
    E:     [B, N, N, d]  edge features
    flags: [B, N] (1 = active, 0 = inactive)
    tol:   tolerance for numerical checks
    """
    B, N, _, d = E.shape

    # Symmetry check
    diff_sym = (E - E.transpose(1, 2)).abs().max().item()
    assert diff_sym < tol, f"E not symmetric, max diff = {diff_sym}"

    # Diagonal zero check
    diag_vals = E.diagonal(dim1=1, dim2=2)  # [B, N, d]
    max_diag = diag_vals.abs().max().item()
    assert max_diag < tol, f"E diagonal not zero, max diag = {max_diag}"

    # Flags masking check
    flags_mask = (flags[:, :, None] * flags[:, None, :]).unsqueeze(-1).float()  # [B, N, N, 1]
    masked = E * (1 - flags_mask)
    max_off = masked.abs().max().item()
    assert max_off < tol, f"E has non-flagged entries not zero, max = {max_off}"
