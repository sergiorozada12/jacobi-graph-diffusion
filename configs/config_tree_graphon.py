from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GeneralConfig:
    seed: int = 185
    use_wandb: bool = True
    save_path: str = "results/"
    device: str = "cuda"
    check_val_every_n_epochs: int = 500
    save_checkpoint_every_n_epochs: int = 1000


@dataclass
class SamplerConfig:
    noise_removal: bool = True
    eps_time: float = 1e-7 # 0.001
    time_schedule: str = "log"
    time_schedule_power: float = 2.0
    snr: float = 0.0
    scale_eps: float = 0.0
    n_steps: int = 1
    num_nodes: int = 20
    test_graphs: int = 100
    use_corrector: bool = False
    predictor: str = "em"  # "em" or "milstein" or "heun"


@dataclass
class DataConfig:
    dir: str = "data"
    data: str = "tree_graphon"
    batch_size: int = 100  # 32
    max_node_num: int = 80
    max_feat_num: int = 1
    test_split: float = 0.2
    val_split: float = 0.1
    init: str = "ones"


@dataclass
class ModelConfig:
    max_feat_num: int = 2
    extra_features_type: str = 'rrwp'
    rrwp_steps: int = 10
    use_sampled_features: bool = True
    n_layers: int = 4
    input_dims: dict = field(default_factory=lambda: {
        "X": 10,
        "E": 10,
        "y": 6,
    })
    hidden_mlp_dims: dict = field(default_factory=lambda: {
        'X': 64,
        'E': 32,
        'y': 64
    })
    hidden_dims: dict = field(default_factory=lambda: {
        'dx': 64,
        'de': 32,
        'dy': 32,
        'n_head': 4,
        'dim_ffX': 64,
        'dim_ffE': 32,
        'dim_ffy': 64
    })
    output_dims: dict = field(default_factory=lambda: {
        "X": 0,
        "E": 2,
        "y": 0,
    })
    score_output_dims: dict = field(default_factory=lambda: {
        "X": 0,
        "E": 1,
        "y": 0,
    })


@dataclass
class TrainConfig:
    lr: float = 0.0002
    amsgrad: bool = True
    weight_decay: float = 1e-12
    eps_time_train: float = 1e-7
    eps_sde_train: float = 1e-7
    time_schedule_train: str = "log"
    time_schedule_power_train: float = 2.0
    num_epochs: int = 30_000
    lambda_train: float = 5.0
    use_ema: bool = True
    ema_decay: float = 0.999
    training_mode: str = "graph"  # options: "graph", "weighted", "direct_score"


@dataclass
class SDEConfig:
    alpha: float = 1.0  # alpha = beta = 1.0 / tested 0.5 / 1.5
    beta: float = 1.0
    num_scales: int = 1000
    s_min: float = 1.0  # s_min = s_max = 1.0 for beta
    s_max: float = 1.0
    order: int = 100
    sample_target: bool = True
    eps_sde: float = 1e-7 # 0.001
    eps_score: float = 1e-10


@dataclass
class MainConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
