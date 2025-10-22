from dataclasses import dataclass, field

@dataclass
class GeneralConfig:
    seed: int = 5 # 58 or 42 or 5 works good
    use_wandb: bool = True
    save_path: str = "results/"
    device: str = "cuda"
    check_val_every_n_epochs: int = 500
    save_checkpoint_every_n_epochs: int = 1000

@dataclass
class SamplerConfig:
    noise_removal: bool = True
    eps_time: float = 1e-3
    snr: float = 0.02 # This goes good with 0.000001 scale_eps = 0
    scale_eps: float = 0.0001 # 0.01 ratio 54
    n_steps: int = 2
    num_nodes: int = 10
    test_graphs: int = 32
    use_corrector: bool = True

@dataclass
class DataConfig:
    dir: str = "data"
    data: str = "planar_baseline"
    batch_size: int = 32
    max_node_num: int = 70
    max_feat_num: int = 1
    test_split: float = 0.2
    val_split: float = 0.1
    init: str = "ones"

@dataclass
class ModelConfig:
    max_feat_num: int = 2
    extra_features_type: str = 'rrwp'
    rrwp_steps: int = 20
    use_sampled_features: bool = True
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
    lr: float = 0.0002 # 0.0002
    amsgrad: bool = True
    weight_decay: float = 1e-12
    eps: float = 1e-5
    num_epochs: int = 20_000
    lambda_train: float = 5.0
    use_ema: bool = False
    ema_decay: float = 0.999

@dataclass
class SDEConfig:
    alpha: float = 1.0
    beta: float = 1.0
    num_scales: int = 200
    s_min: float = 1.0
    s_max: float = 1.0
    order: int = 100 # 30 works good for predictor
    sample_target: bool = False
    eps_sde: float = 1e-2 #3.1622776601e-3 # 3.16 works worse lol
    max_force: float = 1000.0
    eps_score: float = 1e-10
    eps_score_dist: float = 1e-5
    time_schedule: str = "log"
    time_schedule_power: float = 2.0

@dataclass
class MainConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
