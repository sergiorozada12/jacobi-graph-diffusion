import argparse
from omegaconf import OmegaConf

from configs.config_qm9 import MainConfig
from src.parameter_search.tuning import SearchSpace, TuningSettings, run_tuning

def parse_args():
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for QM9.")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = OmegaConf.structured(MainConfig())
    
    # Define a search space relevant for molecular diffusion and Jacobi processes
    search_space = SearchSpace(
        order=[50],
        sample_target=[True],
        eps_sde=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-8, 1e-9],
        eps_score=[1e-10],
        time_schedule=["log"],
        predictor=["em"],
        eps_time=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-8, 1e-9],
        use_corrector=[False], 
        snr=[1.0, 0.1, 0.01, 0.001],
        scale_eps=[1.0, 0.1, 0.01, 0.001],
        n_steps=[1, 2, 5],
    )

    settings = TuningSettings(
        objective="validity", # Relaxed validity
        metric_key=None,
        metrics_alias=None,
        max_trials=None,
        seed=cfg.general.seed,
        device='cuda:0',
        num_graphs=100, # Faster evaluation during tuning
        results_path="results/tuning_qm9.json",
        verbose=True,
        suppress_external_output=True,
        search_space=search_space,
    )

    run_tuning(cfg, settings)

if __name__ == "__main__":
    main()
