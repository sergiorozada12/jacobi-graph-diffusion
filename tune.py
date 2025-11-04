import argparse
from omegaconf import OmegaConf

from configs.config_sbm_2comms import MainConfig
# from configs.config_pa import MainConfig
# from configs.config_tree import MainConfig
# from configs.config_planar import MainConfig
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
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.structured(MainConfig())
    if getattr(cfg.train, "training_mode", "graph") == "direct_score":
        cfg.model.output_dims = dict(cfg.model.score_output_dims)
    
    search_space = SearchSpace(
        order=[100],
        sample_target=[True],
        # eps_sde=[0.00001, 0.000001, 0.0000001, 0.00000001],
        eps_sde=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
        time_schedule=["log"],
        # eps_time=[0.00001, 0.000001, 0.0000001, 0.00000001],
        eps_time=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
        use_corrector=[False],
        snr=[2.0, 1.0, 0.1, 0.01, 0.001, 0.0001],
        scale_eps=[2.0, 1.0, 0.1, 0.01, 0.001, 0.0001],
        n_steps=[1, 2, 5],
    )

    settings = TuningSettings(
        objective="sbm_acc",
        metric_key=None,
        metrics_alias=None,
        max_trials=None,
        seed=None,
        device='cuda:1',
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
