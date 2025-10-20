from omegaconf import OmegaConf

# from configs.config_sbm_2comms import MainConfig
from configs.config_pa import MainConfig
# from configs.config_tree import MainConfig
# from configs.config_planar import MainConfig
from src.parameter_search.tuning import SearchSpace, TuningSettings, run_tuning


def main():
    cfg = OmegaConf.structured(MainConfig())

    search_space = SearchSpace(
        order=[100],
        sample_target=[True],
        eps_sde=[0.00001],
        time_schedule=["log"],
        eps_time=[0.001],
        use_corrector=[True],
        snr=[1.0, 0.1, 0.01, 0.001, 0.0001],
        scale_eps=[1.0, 0.1, 0.01, 0.001, 0.0001],
        n_steps=[1, 2, 5],
    )
    """
    search_space = SearchSpace(
        order=[100],
        sample_target=[True],
        eps_sde=[0.0001],
        time_schedule=["log"],
        eps_time=[0.001],
        use_corrector=[True],
        snr=[1.0, 0.1, 0.01, 0.001, 0.0001],
        scale_eps=[1.0, 0.1, 0.01, 0.001, 0.0001],
        n_steps=[1, 2, 5],
    )
    """

    settings = TuningSettings(
        objective="average_ratio",
        metric_key=None,
        metrics_alias=None,
        max_trials=None,
        seed=None,
        device='cuda:1',
        num_graphs=None,
        results_path="parameters_pa",
        verbose=True,
        suppress_external_output=True,
        search_space=search_space,
    )

    run_tuning(cfg, settings)


if __name__ == "__main__":
    main()
