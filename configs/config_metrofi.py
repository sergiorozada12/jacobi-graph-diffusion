from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GeneralConfig:
    seed: int = 17
    use_wandb: bool = True
    save_path: str = "results/"
    device: str = "cuda"
    check_val_every_n_epochs: int = 100
    save_checkpoint_every_n_epochs: int = 100


@dataclass
class SamplerConfig:
    noise_removal: bool = True
    eps_time: float = 1e-1
    time_schedule: str = "log"
    time_schedule_power: float = 2.0
    snr: float = 0.1
    scale_eps: float = 2.5
    n_steps: int = 5
    num_nodes: int = 70
    test_graphs: int = 100
    use_corrector: bool = True
    predictor: str = "em"  # "em" or "milstein"
    val_use_full_graph: bool = True
    val_keep_isolates: bool = True
    val_use_fixed_nodelist: bool = True


@dataclass
class DataConfig:
    dir: str = "data/metrofi"
    data: str = "metrofi"
    batch_size: int = 100
    max_node_num: int = 70
    max_feat_num: int = 1
    test_split: float = 0.1
    val_split: float = 0.1
    init: str = "ones"
    min_observed_nodes: int = 3
    max_interference: float = 1.0


@dataclass
class ModelConfig:
    max_feat_num: int = 2
    extra_features_type: str = "rrwp"
    rrwp_steps: int = 20
    use_sampled_features: bool = True
    n_layers: int = 8
    input_dims: dict = field(
        default_factory=lambda: {
            "X": 20,
            "E": 20,
            "y": 5 + 1,
        }
    )
    hidden_mlp_dims: dict = field(
        default_factory=lambda: {"X": 128, "E": 64, "y": 128}
    )
    hidden_dims: dict = field(
        default_factory=lambda: {
            "dx": 256,
            "de": 64,
            "dy": 64,
            "n_head": 8,
            "dim_ffX": 256,
            "dim_ffE": 64,
            "dim_ffy": 256,
        }
    )
    output_dims: dict = field(
        default_factory=lambda: {
            "X": 0,
            "E": 2,
            "y": 0,
        }
    )
    score_output_dims: dict = field(
        default_factory=lambda: {
            "X": 0,
            "E": 1,
            "y": 0,
        }
    )


@dataclass
class TrainConfig:
    lr: float = 2e-4
    amsgrad: bool = True
    weight_decay: float = 1e-12
    eps_time_train: float = 1e-1
    eps_sde_train: float = 1e-5
    time_schedule_train: str = "log"
    time_schedule_power_train: float = 2.0
    num_epochs: int = 200
    lambda_train: float = 5.0
    use_ema: bool = True
    ema_decay: float = 0.999
    training_mode: str = "weighted"  # options: "graph", "weighted", "direct_score"


@dataclass
class SDEConfig:
    alpha: float = 1.0
    beta: float = 1.0
    num_scales: int = 200
    s_min: float = 1.0
    s_max: float = 1.0
    order: int = 200
    sample_target: bool = False
    eps_sde: float = 1e-5
    eps_score: float = 1e-10


@dataclass
class MainConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
