import argparse
from omegaconf import OmegaConf
# from configs.config_tree import MainConfig
from configs.config_tree_graphon import MainConfig
from src.parameter_search.tuning import SearchSpace, TuningSettings, run_tuning


def parse_args():
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning with optional custom node-count distribution.")
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=None,
        help="Minimum number of nodes to sample during tuning. Requires --max-nodes.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Maximum number of nodes to sample during tuning. Requires --min-nodes.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Optional checkpoint path to force during tuning (supports Lightning .ckpt and plain .pth).",
    )
    parser.add_argument(
        "--store-name",
        type=str,
        default=None,
        help="Optional hyperparameter store name (writes to hyperparams/<store-name>.json).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.structured(MainConfig())
    if getattr(cfg.train, "training_mode", "graph") == "direct_score":
        cfg.model.output_dims = dict(cfg.model.score_output_dims)
    
    # Previous search space (kept for reference)
    # search_space = SearchSpace(
    #     order=[100],
    #     sample_target=[True],
    #     eps_sde=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    #     eps_score=[1e-10],
    #     time_schedule=["log"],
    #     eps_time=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    #     use_corrector=[False],
    #     snr=[0.001, 0.0001, 0.00001],
    #     scale_eps=[0.01, 0.001, 0.0001, 0.00001],
    #     n_steps=[1, 2],
    # )

    # New OOD-focused search space
    search_space = SearchSpace(
        num_scales=[1000, 1500],
        eps_sde=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        time_schedule=["log", "log_power"],
        time_schedule_power=[0.5, 0.7, 1.0],
        eps_time=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        predictor=["em", "heun", "milstein"],
    )

    settings = TuningSettings(
        objective="tree_acc",
        metric_key=None,
        metrics_alias=None,
        max_trials=None,
        seed=None,
        device='cuda:0',
        num_graphs=None,
        ckpt_path=args.ckpt_path,
        results_path=None,
        store_name=args.store_name,
        verbose=True,
        suppress_external_output=True,
        search_space=search_space,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
    )

    run_tuning(cfg, settings)


if __name__ == "__main__":
    main()
