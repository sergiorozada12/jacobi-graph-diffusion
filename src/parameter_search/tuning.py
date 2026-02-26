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
    WirelessSamplingMetrics,
)
from src.parameter_search.persistence import HyperparamStore
from src.models.transformer_model import GraphTransformer
from src.sample.sampler import Sampler


@dataclass
class SearchSpace:
    order: Optional[List[int]] = None
    sample_target: Optional[List[bool]] = None
    num_scales: Optional[List[int]] = None
    eps_sde: Optional[List[float]] = None
    eps_score: Optional[List[float]] = None
    time_schedule: Optional[List[str]] = None
    time_schedule_power: Optional[List[float]] = None
    eps_time: Optional[List[float]] = None
    predictor: Optional[List[str]] = None
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
    ckpt_path: Optional[Union[str, Path]] = None
    results_path: Optional[Union[str, Path]] = None
    verbose: bool = True
    suppress_external_output: bool = False
    search_space: SearchSpace = field(default_factory=SearchSpace)
    min_nodes: Optional[int] = None
    max_nodes: Optional[int] = None


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


def build_node_distribution(cfg, datamodule, min_nodes: Optional[int], max_nodes: Optional[int]):
    max_nodes_possible = cfg.data.max_node_num + 1
    base_prob = datamodule.node_counts(max_nodes_possible)
    if base_prob.numel() < max_nodes_possible:
        padding = torch.zeros(max_nodes_possible - base_prob.numel(), dtype=base_prob.dtype)
        base_prob = torch.cat([base_prob, padding], dim=0)

    if min_nodes is None and max_nodes is None:
        return DistributionNodes(prob=base_prob)

    if min_nodes is None or max_nodes is None:
        raise ValueError("Both min_nodes and max_nodes must be provided.")
    if min_nodes < 1:
        raise ValueError("Minimum number of nodes must be at least 1.")
    if max_nodes < min_nodes:
        raise ValueError("Maximum number of nodes must be greater than or equal to the minimum.")

    max_node_num = cfg.data.max_node_num
    if max_nodes > max_node_num:
        raise ValueError(
            f"Requested max_nodes={max_nodes} exceeds cfg.data.max_node_num={max_node_num}. "
            "Increase the configuration limits before building the distribution."
        )

    probs = torch.zeros_like(base_prob)
    probs[min_nodes : max_nodes + 1] = 1.0
    total_mass = probs.sum()
    if not torch.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("Probability mass over the requested range is zero; verify the node bounds.")
    probs /= total_mass
    return DistributionNodes(prob=probs)


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
        ("metrofi", "metrofi"),
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
        "metrofi": WirelessSamplingMetrics,
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


def _load_graph_transformer_state_dict(raw_ckpt: Dict[str, Any], use_ema: bool) -> Dict[str, Any]:
    # Lightning checkpoints typically store weights under `state_dict` with prefixes
    # like `model.` (or `ema_model.` when EMA is enabled).
    if "state_dict" in raw_ckpt and isinstance(raw_ckpt["state_dict"], dict):
        pl_state = raw_ckpt["state_dict"]
        model_state = {k[len("model."):]: v for k, v in pl_state.items() if k.startswith("model.")}
        ema_state = {k[len("ema_model."):]: v for k, v in pl_state.items() if k.startswith("ema_model.")}

        if use_ema and ema_state:
            return ema_state
        if model_state:
            return model_state
        return pl_state

    return raw_ckpt


def load_weights(
    cfg,
    model: torch.nn.Module,
    use_ema: bool = True,
    ckpt_path: Optional[Union[str, Path]] = None,
) -> None:
    if ckpt_path is not None:
        weight_path = Path(ckpt_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"Provided checkpoint path does not exist: {weight_path}")
    else:
        ckpt_dir = Path("checkpoints") / cfg.data.data
        ema_path = ckpt_dir / "weights_ema.pth"
        weights_path = ckpt_dir / "weights.pth"
        weight_path = ema_path if use_ema and ema_path.exists() else weights_path
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Could not find checkpoint weights for dataset '{cfg.data.data}'. "
                f"Searched: {ema_path} and {weights_path}"
            )

    raw = torch.load(weight_path, map_location="cpu")
    state_dict = _load_graph_transformer_state_dict(raw, use_ema=use_ema) if isinstance(raw, dict) else raw
    model.load_state_dict(state_dict)


def prepare_search_space(cfg, space: SearchSpace):
    orders = space.order or [cfg.sde.order]
    sample_targets = space.sample_target or [cfg.sde.sample_target]
    num_scales_list = space.num_scales or [cfg.sde.num_scales]
    eps_sde_list = space.eps_sde or [cfg.sde.eps_sde]
    eps_score_list = space.eps_score or [cfg.sde.eps_score]
    time_schedules = space.time_schedule or [cfg.sde.time_schedule]
    time_schedule_powers = space.time_schedule_power or [cfg.sde.time_schedule_power]
    eps_time_list = space.eps_time or [cfg.sampler.eps_time]
    predictors = space.predictor or [getattr(cfg.sampler, "predictor", "em")]
    use_correctors = space.use_corrector or [cfg.sampler.use_corrector]
    snr_list = space.snr or [cfg.sampler.snr]
    scale_eps_list = space.scale_eps or [cfg.sampler.scale_eps]
    n_steps_list = space.n_steps or [cfg.sampler.n_steps]

    return (
        orders,
        sample_targets,
        num_scales_list,
        eps_sde_list,
        eps_score_list,
        time_schedules,
        time_schedule_powers,
        eps_time_list,
        predictors,
        use_correctors,
        snr_list,
        scale_eps_list,
        n_steps_list,
    )


def enumerate_trials(
    orders: Iterable[int],
    sample_targets: Iterable[bool],
    num_scales_list: Iterable[int],
    eps_sde_list: Iterable[float],
    eps_score_list: Iterable[float],
    time_schedules: Iterable[str],
    time_schedule_powers: Iterable[float],
    eps_time_list: Iterable[float],
    predictors: Iterable[str],
    use_correctors: Iterable[bool],
    snr_list: Iterable[float],
    scale_eps_list: Iterable[float],
    n_steps_list: Iterable[int],
) -> List[Dict[str, Any]]:
    schedule_power_values = list(time_schedule_powers)
    if not schedule_power_values:
        raise ValueError("time_schedule_power search space must not be empty.")

    trials = []
    base_space = product(
        orders,
        sample_targets,
        num_scales_list,
        eps_sde_list,
        eps_score_list,
        time_schedules,
        eps_time_list,
        predictors,
    )
    schedules_using_power = {"power", "log_power", "double_log"}

    for order, sample_target, num_scales, eps_sde, eps_score, time_schedule, eps_time, predictor in base_space:
        active_powers = (
            schedule_power_values
            if str(time_schedule).lower() in schedules_using_power
            else [schedule_power_values[0]]
        )
        for time_schedule_power in active_powers:
            for use_corrector in use_correctors:
                if use_corrector:
                    for snr, scale_eps, n_steps in product(snr_list, scale_eps_list, n_steps_list):
                        trials.append(
                            {
                                "order": order,
                                "sample_target": sample_target,
                                "num_scales": num_scales,
                                "eps_sde": eps_sde,
                                "eps_score": eps_score,
                                "time_schedule": time_schedule,
                                "time_schedule_power": time_schedule_power,
                                "eps_time": eps_time,
                                "predictor": predictor,
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
                            "num_scales": num_scales,
                            "eps_sde": eps_sde,
                            "eps_score": eps_score,
                            "time_schedule": time_schedule,
                            "time_schedule_power": time_schedule_power,
                            "eps_time": eps_time,
                            "predictor": predictor,
                            "use_corrector": False,
                        }
                    )
    return trials


def configure_trial(cfg, params: Dict[str, Any]):
    cfg.sde.order = params["order"]
    cfg.sde.sample_target = params["sample_target"]
    cfg.sde.num_scales = params["num_scales"]
    cfg.sde.eps_sde = params["eps_sde"]
    cfg.sde.eps_score = params["eps_score"]
    cfg.sde.time_schedule = params["time_schedule"]
    cfg.sde.time_schedule_power = params["time_schedule_power"]
    cfg.sampler.eps_time = params["eps_time"]
    cfg.sampler.predictor = params["predictor"]
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
    if settings.min_nodes is not None or settings.max_nodes is not None:
        if settings.min_nodes is None or settings.max_nodes is None:
            raise ValueError("Provide both min_nodes and max_nodes or neither.")
        if settings.max_nodes > cfg.data.max_node_num:
            cfg.data.max_node_num = settings.max_nodes
        if settings.max_nodes > cfg.sampler.num_nodes:
            cfg.sampler.num_nodes = settings.max_nodes

    pl.seed_everything(cfg.general.seed, workers=True)

    if "metrofi" in cfg.data.data.lower():
        from src.dataset.wireless import WirelessDatasetModule
        datamodule = WirelessDatasetModule(cfg)
    else:
        datamodule = SpectreDatasetModule(cfg)
    with maybe_silence(settings.suppress_external_output and not settings.verbose):
        datamodule.setup()

    metrics_alias = resolve_metrics_name(cfg.data.data, settings.metrics_alias)
    metrics_cls = resolve_metrics_class(metrics_alias)

    metrics_for_ref = metrics_cls(datamodule)
    with maybe_silence(settings.suppress_external_output and not settings.verbose):
        ref_metrics = compute_reference_metrics(datamodule, metrics_for_ref)
    metrics_module = metrics_cls(datamodule)

    node_dist = build_node_distribution(cfg, datamodule, settings.min_nodes, settings.max_nodes)

    model = build_model(cfg)
    load_weights(cfg, model, use_ema=cfg.train.use_ema, ckpt_path=settings.ckpt_path)
    model.eval()

    search_space = prepare_search_space(cfg, settings.search_space)
    all_trials = enumerate_trials(*search_space)
    all_trials = maybe_limit_trials(all_trials, settings.max_trials, cfg.general.seed)

    if not all_trials:
        raise ValueError("No trials to run. Please specify a non-empty search space.")

    if settings.verbose:
        print(f"Running {len(all_trials)} trial(s) with objective '{settings.objective}'.")

    minimize_objective = settings.objective == "average_ratio"
    objective_mode = "min" if minimize_objective else "max"
    if settings.min_nodes is not None and settings.max_nodes is not None:
        dataset_id = f"{cfg.data.data}_nodes_{settings.min_nodes}_{settings.max_nodes}"
    else:
        dataset_id = cfg.data.data
    store = HyperparamStore(dataset_id, settings.objective, objective_mode)
    store.start_run()

    best_result = None
    best_metrics = None
    best_objective_value = math.inf if minimize_objective else -math.inf
    history_best = store.best_history()
    if history_best:
        best_result = history_best.get("params")
        best_metrics = history_best.get("metrics")
        best_objective_value = history_best.get("objective", best_objective_value)

    best_run_record = None
    best_run_value = math.inf if minimize_objective else -math.inf
    trial_results: List[Dict[str, Any]] = []

    try:
        for idx, params in enumerate(all_trials, start=1):
            if settings.verbose:
                print(f"\n--- Trial {idx}/{len(all_trials)} ---")
                print(f"Parameters: {params}")

            if store.has_successful_trial(params):
                existing = store.get_trial(params)
                if settings.verbose:
                    recorded_value = existing.get("objective") if existing else None
                    print(
                        "Skipping trial (already recorded with status 'ok'). "
                        f"Stored objective: {recorded_value}"
                    )
                trial_results.append(
                    {
                        "params": params,
                        "status": "skipped",
                        "objective": existing.get("objective") if existing else None,
                        "metrics": existing.get("metrics") if existing else None,
                    }
                )
                continue

            trial_seed = cfg.general.seed + idx
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
                        trial_seed=trial_seed,
                    )
                obj_value, objective_key = compute_objective(
                    trial_metrics,
                    settings.objective,
                    settings.metric_key,
                )
            except Exception as exc:
                print(f"Trial {idx} failed: {exc}")
                record = store.record_failure(
                    params,
                    error=str(exc),
                    seed=trial_seed,
                    trial_index=idx,
                )
                trial_results.append(
                    {"params": params, "status": "failed", "error": record["error"]}
                )
                continue

            if settings.verbose:
                print(f"Objective ({objective_key}) = {obj_value}")

            record = store.record_success(
                params,
                objective_value=obj_value,
                metrics=trial_metrics,
                objective_key=objective_key,
                seed=trial_seed,
                trial_index=idx,
            )

            trial_results.append(
                {"params": params, "status": "ok", "objective": obj_value, "metrics": trial_metrics}
            )

            improved = False
            if minimize_objective:
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

            better_run = False
            if minimize_objective:
                if obj_value < best_run_value:
                    better_run = True
            else:
                if obj_value > best_run_value:
                    better_run = True

            if better_run:
                best_run_value = obj_value
                best_run_record = record

            if not settings.verbose:
                best_display = best_objective_value if best_result is not None else obj_value
                print(
                    f"Trial {idx}/{len(all_trials)} | params={params} | "
                    f"{objective_key}={obj_value:g} | best={best_display:g}"
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        store.finalize_run(best_run_record)

    if best_result is None:
        raise RuntimeError("All trials failed. Check logs above for details.")

    print("\n=== Best configuration ===")
    print(json.dumps({"params": best_result, "objective": best_objective_value}, indent=2))
    print("\nAssociated metrics:")
    print(json.dumps(best_metrics, indent=2))

    best_history_snapshot = store.best_history()
    best_last_run_snapshot = store.best_last_run()

    if settings.results_path:
        results_path = Path(settings.results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w") as f:
            json.dump(
                {
                    "objective": settings.objective,
                    "trials": trial_results,
                    "best_history": best_history_snapshot,
                    "best_run": best_last_run_snapshot,
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
