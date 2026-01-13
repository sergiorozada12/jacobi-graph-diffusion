import argparse
from omegaconf import OmegaConf

from configs.config_metrofi import MainConfig
from src.parameter_search.tuning import SearchSpace, TuningSettings, run_tuning


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for MetroFi (wireless) sampler settings.")
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
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.structured(MainConfig())

    # Use score-style output dims when training_mode requests it.
    if getattr(cfg.train, "training_mode", "graph") == "direct_score":
        cfg.model.output_dims = dict(cfg.model.score_output_dims)

    search_space = SearchSpace(
        order=[cfg.sde.order],
        sample_target=[cfg.sde.sample_target],
        eps_sde=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        eps_score=[1e-8, 1e-9, 1e-10],
        time_schedule=[cfg.sde.time_schedule],
        eps_time=[1e-2, 1e-3, 1e-4, 1e-5],
        use_corrector=[False],
        snr=[0.02, 0.01, 0.005],
        scale_eps=[1e-3, 1e-4, 1e-5],
        n_steps=[1, 2],
    )

    settings = TuningSettings(
        objective="average_ratio",
        metric_key=None,
        metrics_alias="metrofi",
        max_trials=None,
        seed=None,
        device=cfg.general.device,
        num_graphs=None,
        results_path=None,
        verbose=True,
        suppress_external_output=True,
        search_space=search_space,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
    )

    run_tuning(cfg, settings)


if __name__ == "__main__":
    main()
