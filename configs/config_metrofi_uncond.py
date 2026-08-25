from dataclasses import dataclass, field
from typing import Optional

from configs.config_metrofi_cond import (
    DataConfig,
    GeneralConfig as BaseGeneralConfig,
    ModelConfig as BaseModelConfig,
    SamplerConfig as BaseSamplerConfig,
    SDEConfig,
    TrainConfig as BaseTrainConfig,
)


@dataclass
class GeneralConfig(BaseGeneralConfig):
    name: Optional[str] = "metrofi-uncond-pe-nosf"


@dataclass
class SamplerConfig(BaseSamplerConfig):
    guidance_scale: float = 0.0


@dataclass
class ModelConfig(BaseModelConfig):
    conditional: bool = False
    use_location_condition: bool = False
    condition_dim: int = 0


@dataclass
class TrainConfig(BaseTrainConfig):
    condition_dropout_prob: float = 0.0


@dataclass
class MainConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
