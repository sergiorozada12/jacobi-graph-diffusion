from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class GeneralConfig:
    seed: int = 42
    use_wandb: bool = True
    save_path: str = "results/"
    device: str = "cuda"

@dataclass
class SamplerConfig:
    noise_removal: bool = True
    eps: float = 1e-4
    snr: float = 0.01
    scale_eps: float = 1.0
    n_steps: int = 2
    num_nodes: int = 20

@dataclass
class DataConfig:
    dir: str = "data"
    data: str = "tree"
    batch_size: int = 128
    max_node_num: int = 20
    max_feat_num: int = 1
    test_split: float = 0.2
    val_split: float = 0.1
    init: str = "ones"

@dataclass
class ModelConfig:
    max_feat_num: int = 2
    nhid: int = 32
    num_layers: int = 7
    num_linears: int = 2
    c_init: int = 2
    c_hid: int = 8
    c_final: int = 4
    adim: int = 32
    num_heads: int = 4
    conv: str = "GCN"

@dataclass
class TrainConfig:
    lr: float = 0.01
    weight_decay: float = 0.0001
    eps: float = 1e-5
    lr_schedule: bool = True
    lr_decay: float = 0.999
    num_epochs: int = 5_000
    grad_norm: float = 1.0
    lambda_adj: float = 1.0
    lambda_x: float = 0.1
    features: List[str] = field(default_factory=lambda: ["degree"])
    k_eig: Optional[int] = None

@dataclass
class SDEConfig:
    alpha: float = 1.0
    beta: float = 1.0
    num_scales: int = 1000
    speed: float = 1.0
    order: int = 10

@dataclass
class MainConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)