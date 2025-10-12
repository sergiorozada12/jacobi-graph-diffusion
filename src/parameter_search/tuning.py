import importlib
import io
import json
import math
import random
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from src.dataset.spectre import SpectreDatasetModule
from src.dataset.utils import DistributionNodes, compute_reference_metrics
from src.metrics.val import (
    EgoSamplingMetrics,
    IMDBSamplingMetrics,
    PASamplingMetrics,
    PlanarSamplingMetrics,
    ProteinSamplingMetrics,
    SBMSamplingMetrics,
    SpectreSamplingMetrics,
    TreeSamplingMetrics,
)
from src.models.transformer_model import GraphTransformer
from src.sample.sampler import Sampler


@dataclass
class SearchSpace:
    order: Optional[List[int]] = None
    sample_target: Optional[List[bool]] = None
    eps_sde: Optional[List[float]] = None
    time_schedule: Optional[List[str]] = None
    eps_time: Optional[List[float]] = None
    use_corrector: Optional[List[bool]] = None
    snr: Optional[List[float]] = None
    scale_eps: Optional[List[float]] = None
    n_steps: Optional[List[int]] = None


@dataclass
class TuningSettings:
    objective: str = "average_ratio"
    metric_key: Optional[str] = None
    metrics_alias: Optional[str] = None
    max_trials: Optional[int] = None
    seed: Optional[int] = None
    device: Optional[str] = None
    num_graphs: Optional[int] = None
    results_path: Optional[Union[str, Path]] = None
    verbose: bool = True
    suppress_external_output: bool = False
    search_space: SearchSpace = field(default_factory=SearchSpace)


def load_main_config(module_path: str, class_name: str = "MainConfig"):
    module = importlib.import_module(module_path)
    if not hasattr(module, class_name):
        raise AttributeError(
            f"Module '{module_path}' does not define '{class_name}'. "
            "Provide a different config class name if needed."
        )
    config_cls = getattr(module, class_name)
    return config_cls()


def clone_config(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cloned = OmegaConf.create(cfg_dict)
    OmegaConf.set_struct(cloned, False)
    return cloned


def resolve_metrics_name(dataset_name: str, override: Optional[str]) -> str:
    if override:
        return override.lower()

    name = dataset_name.lower()
    mapping = [
        ("sbm", "sbm"),
        ("planar", "planar"),
        ("tree", "tree"),
        ("pa", "pa"),
        ("ego", "ego"),
        ("protein", "protein"),
        ("imdb", "imdb"),
    ]
    for pattern, alias in mapping:
        if pattern in name:
            return alias

    return "sbm"


def resolve_metrics_class(alias: str) -> Type[SpectreSamplingMetrics]:
    metrics_map = {
        "sbm": SBMSamplingMetrics,
        "planar": PlanarSamplingMetrics,
        "tree": TreeSamplingMetrics,
        "pa": PASamplingMetrics,
        "ego": EgoSamplingMetrics,
        "protein": ProteinSamplingMetrics,
        "imdb": IMDBSamplingMetrics,
    }
    if alias not in metrics_map:
        raise ValueError(
            f"Unknown metrics alias '{alias}'. "
            f"Supported options: {', '.join(metrics_map.keys())}"
        )
    return metrics_map[alias]


def get_accuracy_key(metrics: Dict[str, float], explicit_key: Optional[str]) -> str:
    if explicit_key:
        if explicit_key not in metrics:
            raise ValueError(f"Metric '{explicit_key}' not present in results: {list(metrics)}")
        return explicit_key

    for key in ("sbm_acc", "planar_acc", "tree_acc", "pa_acc"):
        if key in metrics:
            return key

    candidates = [k for k in metrics if k.endswith("_acc")]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(
            "Multiple accuracy-like metrics found. "
            "Please choose one explicitly via metric_key."
        )
    raise ValueError(
        "No accuracy metric found in results. "
        "Use objective 'average_ratio' or specify metric_key explicitly."
    )


def build_model(cfg):
    model = GraphTransformer(
        n_layers=cfg.model.n_layers,
        input_dims=cfg.model.input_dims,
        hidden_mlp_dims=cfg.model.hidden_mlp_dims,
        hidden_dims=cfg.model.hidden_dims,
        output_dims=cfg.model.output_dims,
        act_fn_in=torch.nn.ReLU(),
        act_fn_out=torch.nn.ReLU(),
    )
    return model


def load_weights(cfg, model: torch.nn.Module, use_ema: bool = True) -> None:
    ckpt_dir = Path("checkpoints") / cfg.data.data
    ema_path = ckpt_dir / "weights_ema.pth"
    weights_path = ckpt_dir / "weights.pth"

    weight_path = ema_path if use_ema and ema_path.exists() else weights_path
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Could not find checkpoint weights for dataset '{cfg.data.data}'. "
            f"Searched: {ema_path} and {weights_path}"
        )
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)


def prepare_search_space(cfg, space: SearchSpace):
    orders = space.order or [cfg.sde.order]
    sample_targets = space.sample_target or [cfg.sde.sample_target]
    eps_sde_list = space.eps_sde or [cfg.sde.eps_sde]
    time_schedules = space.time_schedule or [cfg.sde.time_schedule]
    eps_time_list = space.eps_time or [cfg.sampler.eps_time]
    use_correctors = space.use_corrector or [cfg.sampler.use_corrector]
    snr_list = space.snr or [cfg.sampler.snr]
    scale_eps_list = space.scale_eps or [cfg.sampler.scale_eps]
    n_steps_list = space.n_steps or [cfg.sampler.n_steps]

    return (
        orders,
        sample_targets,
        eps_sde_list,
        time_schedules,
        eps_time_list,
        use_correctors,
        snr_list,
        scale_eps_list,
        n_steps_list,
    )


def enumerate_trials(
    orders: Iterable[int],
    sample_targets: Iterable[bool],
    eps_sde_list: Iterable[float],
    time_schedules: Iterable[str],
    eps_time_list: Iterable[float],
    use_correctors: Iterable[bool],
    snr_list: Iterable[float],
    scale_eps_list: Iterable[float],
    n_steps_list: Iterable[int],
) -> List[Dict[str, Any]]:
    base_space = product(orders, sample_targets, eps_sde_list, time_schedules, eps_time_list)
    trials = []
    for order, sample_target, eps_sde, time_schedule, eps_time in base_space:
        for use_corrector in use_correctors:
            if use_corrector:
                for snr, scale_eps, n_steps in product(snr_list, scale_eps_list, n_steps_list):
                    trials.append(
                        {
                            "order": order,
                            "sample_target": sample_target,
                            "eps_sde": eps_sde,
                            "time_schedule": time_schedule,
                            "eps_time": eps_time,
                            "use_corrector": True,
                            "snr": snr,
                            "scale_eps": scale_eps,
                            "n_steps": n_steps,
                        }
                    )
            else:
                trials.append(
                    {
                        "order": order,
                        "sample_target": sample_target,
                        "eps_sde": eps_sde,
                        "time_schedule": time_schedule,
                        "eps_time": eps_time,
                        "use_corrector": False,
                    }
                )
    return trials


def configure_trial(cfg, params: Dict[str, Any]):
    cfg.sde.order = params["order"]
    cfg.sde.sample_target = params["sample_target"]
    cfg.sde.eps_sde = params["eps_sde"]
    cfg.sde.time_schedule = params["time_schedule"]
    cfg.sampler.eps_time = params["eps_time"]
    cfg.sampler.use_corrector = params["use_corrector"]

    if cfg.sampler.use_corrector:
        cfg.sampler.snr = params["snr"]
        cfg.sampler.scale_eps = params["scale_eps"]
        cfg.sampler.n_steps = params["n_steps"]


def evaluate_trial(
    base_cfg,
    params: Dict[str, Any],
    model: torch.nn.Module,
    datamodule: SpectreDatasetModule,
    node_dist: Optional[DistributionNodes],
    metrics_module: SpectreSamplingMetrics,
    ref_metrics: Dict[str, Dict[str, float]],
    trial_seed: int,
) -> Dict[str, float]:
    trial_cfg = clone_config(base_cfg)
    configure_trial(trial_cfg, params)

    pl.seed_everything(trial_seed, workers=True)

    sampler = Sampler(cfg=trial_cfg, model=model, node_dist=node_dist)
    sampler.model.eval()

    with torch.inference_mode():
        samples, _ = sampler.sample()

    metrics_module.reset()
    metrics = metrics_module.forward(
        samples,
        ref_metrics=ref_metrics,
        local_rank=0,
        test=True,
    )
    return metrics


def compute_objective(
    metrics: Dict[str, float],
    objective: str,
    metric_key: Optional[str] = None,
) -> Tuple[float, str]:
    if objective == "average_ratio":
        if "average_ratio" not in metrics or metrics["average_ratio"] < 0:
            raise ValueError(
                "average_ratio not available for this trial. "
                "Ensure ratio metrics were computed."
            )
        return metrics["average_ratio"], "average_ratio"

    if objective == "accuracy":
        key = get_accuracy_key(metrics, metric_key)
        return metrics[key], key

    if objective in metrics:
        return metrics[objective], objective

    raise ValueError(f"Objective '{objective}' not found in metrics {list(metrics)}.")


def maybe_limit_trials(
    trials: List[Dict[str, Any]],
    max_trials: Optional[int],
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    if max_trials is None or max_trials >= len(trials):
        return trials
    rng = random.Random(seed)
    rng.shuffle(trials)
    return trials[:max_trials]


@contextmanager
def maybe_silence(condition: bool):
    if condition:
        with redirect_stdout(io.StringIO()):
            yield
    else:
        yield


def run_tuning(base_cfg, settings: TuningSettings):
    cfg = clone_config(base_cfg)

    if settings.device:
        cfg.general.device = settings.device
    if settings.num_graphs is not None:
        cfg.sampler.test_graphs = settings.num_graphs
    if settings.seed is not None:
        cfg.general.seed = settings.seed

    pl.seed_everything(cfg.general.seed, workers=True)

    datamodule = SpectreDatasetModule(cfg)
    with maybe_silence(settings.suppress_external_output and not settings.verbose):
        datamodule.setup()

    metrics_alias = resolve_metrics_name(cfg.data.data, settings.metrics_alias)
    metrics_cls = resolve_metrics_class(metrics_alias)

    metrics_for_ref = metrics_cls(datamodule)
    with maybe_silence(settings.suppress_external_output and not settings.verbose):
        ref_metrics = compute_reference_metrics(datamodule, metrics_for_ref)
    metrics_module = metrics_cls(datamodule)

    node_dist = DistributionNodes(prob=datamodule.node_counts())

    model = build_model(cfg)
    load_weights(cfg, model, use_ema=cfg.train.use_ema)
    model.eval()

    search_space = prepare_search_space(cfg, settings.search_space)
    all_trials = enumerate_trials(*search_space)
    all_trials = maybe_limit_trials(all_trials, settings.max_trials, cfg.general.seed)

    if not all_trials:
        raise ValueError("No trials to run. Please specify a non-empty search space.")

    if settings.verbose:
        print(f"Running {len(all_trials)} trial(s) with objective '{settings.objective}'.")

    best_result = None
    best_metrics = None
    best_objective_value = math.inf if settings.objective == "average_ratio" else -math.inf
    trial_results = []

    for idx, params in enumerate(all_trials, start=1):
        if settings.verbose:
            print(f"\n--- Trial {idx}/{len(all_trials)} ---")
            print(f"Parameters: {params}")

        try:
            with maybe_silence(settings.suppress_external_output and not settings.verbose):
                trial_metrics = evaluate_trial(
                    cfg,
                    params,
                    model,
                    datamodule,
                    node_dist,
                    metrics_module,
                    ref_metrics,
                    trial_seed=cfg.general.seed + idx,
                )
            obj_value, objective_key = compute_objective(
                trial_metrics,
                settings.objective,
                settings.metric_key,
            )
        except Exception as exc:
            print(f"Trial {idx} failed: {exc}")
            trial_results.append(
                {"params": params, "status": "failed", "error": str(exc)}
            )
            continue

        if settings.verbose:
            print(f"Objective ({objective_key}) = {obj_value}")
        trial_results.append(
            {"params": params, "status": "ok", "objective": obj_value, "metrics": trial_metrics}
        )

        improved = False
        if settings.objective == "average_ratio":
            if obj_value < best_objective_value:
                improved = True
        else:
            if obj_value > best_objective_value:
                improved = True

        if improved:
            best_objective_value = obj_value
            best_result = params
            best_metrics = trial_metrics
            if settings.verbose:
                print("New best configuration found.")

        if not settings.verbose:
            best_display = best_objective_value if best_result is not None else obj_value
            print(
                f"Trial {idx}/{len(all_trials)} | params={params} | "
                f"{objective_key}={obj_value:g} | best={best_display:g}"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_result is None:
        raise RuntimeError("All trials failed. Check logs above for details.")

    print("\n=== Best configuration ===")
    print(json.dumps({"params": best_result, "objective": best_objective_value}, indent=2))
    print("\nAssociated metrics:")
    print(json.dumps(best_metrics, indent=2))

    if settings.results_path:
        results_path = Path(settings.results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w") as f:
            json.dump(
                {
                    "objective": settings.objective,
                    "trials": trial_results,
                    "best": {
                        "params": best_result,
                        "objective": best_objective_value,
                        "metrics": best_metrics,
                    },
                },
                f,
                indent=2,
            )
        print(f"\nSaved detailed results to {results_path}")

    return best_result, best_objective_value, best_metrics
