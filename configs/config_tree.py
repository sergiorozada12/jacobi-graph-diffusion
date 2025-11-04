from dataclasses import dataclass, field

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
    eps_time: float = 0.001
    snr: float = 0.1
    scale_eps: float = 0.1
    n_steps: int = 5
    num_nodes: int = 20
    test_graphs: int = 16
    use_corrector: bool = False

@dataclass
class DataConfig:
    dir: str = "data"
    data: str = "tree_baseline"
    batch_size: int = 16 # 32
    max_node_num: int = 80
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
    n_layers: int = 10
    input_dims: dict = field(default_factory=lambda: {
        "X": 20,    # rrwp_steps
        "E": 20,    # rrwp_steps + distribution 20 + 2
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
    eps: float = 1e-4
    num_epochs: int = 5_000
    lambda_train: float = 5.0
    use_ema: bool = True
    ema_decay: float = 0.999
    training_mode: str = "graph"  # options: "graph", "direct_score"
    score_loss_weight: float = 1.0

@dataclass
class SDEConfig:
    alpha: float = 1.0 # alpha = beta = 1.0 / tested 0.5 / 1.5
    beta: float = 1.0
    num_scales: int = 1000
    s_min: float = 1.0 # s_min = s_max = 1.0 for beta
    s_max: float = 1.0
    order: int = 100
    sample_target: bool = True
    eps_sde: float = 0.001
    max_force: float = 1000.0
    eps_score: float = 1e-10
    eps_score_dist: float = 1e-5
    time_schedule: str = "log" # This linear for beta
    time_schedule_power: float = 2.0

@dataclass
class MainConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
