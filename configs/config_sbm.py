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
    snr: float = 0.0001
    scale_eps: float = 1.0
    n_steps: int = 1
    num_nodes: int = 40

@dataclass
class DataConfig:
    dir: str = "data"
    data: str = "sbm"
    batch_size: int = 64
    max_node_num: int = 40
    max_feat_num: int = 1
    test_split: float = 0.2
    val_split: float = 0.1
    init: str = "ones"

@dataclass
class ModelConfig:
    max_feat_num: int = 2
    extra_features_type: str = 'rrwp'
    rrwp_steps: int = 20
    n_layers: int = 8
    input_dims: dict = field(default_factory=lambda: {
        "X": 20,    # rrwp_steps
        "E": 20,    # rrwp_steps
        "y": 5 + 1, # +1 for time conditioning
    })
    hidden_mlp_dims: dict = field(default_factory=lambda: {
        'X': 128,
        'E': 64,
        'y': 128
    })
    hidden_dims: dict = field(default_factory=lambda: {
        'dx': 256,
        'de': 64,
        'dy': 64,
        'n_head': 8,
        'dim_ffX': 256,
        'dim_ffE': 64,
        'dim_ffy': 256
    })
    output_dims: dict = field(default_factory=lambda: {
        "X": 0,
        "E": 2,
        "y": 0,
    })

@dataclass
class TrainConfig:
    lr: float = 0.0002
    amsgrad: bool = True
    weight_decay: float = 1e-12
    eps: float = 1e-5
    num_epochs: int = 5_000
    lambda_train: float = 5.0

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