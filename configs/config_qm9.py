from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GeneralConfig:
    seed: int = 42
    use_wandb: bool = True
    save_path: str = "results/"
    device: str = "cuda"
    check_val_every_n_epochs: int = 50
    save_checkpoint_every_n_epochs: int = 100

@dataclass
class SamplerConfig:
    noise_removal: bool = True
    eps_time: float = 1e-3
    time_schedule: str = "cosine"
    time_schedule_power: float = 2.0
    n_steps: int = 100
    predictor: str = "em" 
    sample_X: bool = True
    sample_E: bool = True
    num_nodes: int = 9
    snr: float = 1.0
    scale_eps: float = 2.0
    use_corrector: bool = False
    test_graphs: Optional[int] = 50

@dataclass
class DataConfig:
    dir: str = "data/qm9"
    data: str = "qm9"
    batch_size: int = 128
    max_node_num: int = 9
    max_feat_num: int = 4 # C, N, O, F
    init: str = "zeros"

@dataclass
class ModelConfig:
    n_layers: int = 5
    extra_features_type: str = "rrwp"
    rrwp_steps: int = 12
    hidden_mlp_dims: dict = field(default_factory=lambda: {"X": 256, "E": 128, "y": 128})
    hidden_dims: dict = field(
        default_factory=lambda: {
            "dx": 256,
            "de": 64,
            "dy": 64,
            "n_head": 8,
            "dim_ffX": 256,
            "dim_ffE": 128,
            "dim_ffy": 128,
        }
    )
    input_dims: dict = field(default_factory=lambda: {
        "X": 17,    # 3 (v_X_t, K_X-1 stick-breaking) + 12 (RRWP) + 2 (Molecular)
        "E": 16,    # 4 (v_E_t, K_E-1 stick-breaking) + 12 (RRWP)
        "y": 7      # 1 (n) + 4 (Cycles) + 1 (Weight) + 1 (t)
    })
    output_dims: dict = field(default_factory=lambda: {"X": 4, "E": 5, "y": 0}) # K_X=4 (C,N,O,F)
    activation: str = "relu"

@dataclass
class DatasetConfig:
    name: str = "qm9"
    datadir: str = "data/qm9/qm9_pyg"
    node_n_types: int = 4   # C, N, O, F (no H, since remove_h=True)
    edge_n_types: int = 5   # no-bond, single, double, triple, aromatic
    y_n_types: int = 0
    max_node_num: int = 9

@dataclass
class TrainConfig:
    lr: float = 2e-4
    amsgrad: bool = True
    weight_decay: float = 1e-12
    eps_time_train: float = 1e-5
    eps_sde_train: float = 1e-1
    time_schedule_train: str = "cosine"
    time_schedule_power_train: float = 2.0
    num_epochs: int = 1000
    lambda_train: float = 1.0 # total multiplier
    lambda_node: float = 1.0
    lambda_edge: float = 10.0 # Edges are harder
    use_ema: bool = True
    ema_decay: float = 0.999
    training_mode: str = "graph" # Data prediction mode

@dataclass
class SDEConfig:
    type: str = "stick_breaking"
    alpha: float = 1.0
    beta: float = 1.0
    num_scales: int = 1000
    s_min: float = 1.0
    s_max: float = 1.0
    order: int = 30
    sample_target: bool = True
    eps_sde: float = 1e-3
    eps_score: float = 1e-10

@dataclass
class MainConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
