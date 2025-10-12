from omegaconf import OmegaConf

from configs.config_sbm_2comms import MainConfig
from configs.config_pa import MainConfig
from src.parameter_search.tuning import SearchSpace, TuningSettings, run_tuning


def main():
    cfg = OmegaConf.structured(MainConfig())

    # Adjust the per-parameter lists below to explore different settings.
    search_space = SearchSpace(
        order=[85, 90, 95],
        sample_target=[False],
        eps_sde=[0.0002, 0.00025, 0.0003],
        time_schedule=["log"],
        eps_time=[0.00325, 0.0035, 0.00375],
        use_corrector=[True],
        snr=[0.675, 0.7, 0.725],
        scale_eps=[0.225, 0.25, 0.275],
        n_steps=[1],
    )

    settings = TuningSettings(
        objective="average_ratio",
        metric_key=None,
        metrics_alias=None,
        max_trials=None,
        seed=None,
        device=None,
        num_graphs=None,
        results_path="parameters_pa_7",
        verbose=True,
        suppress_external_output=True,
        search_space=search_space,
    )

    run_tuning(cfg, settings)


if __name__ == "__main__":
    main()

"""
=== Best configuration ===
{
  "params": {
    "order": 100,
    "sample_target": false,
    "eps_sde": 0.01,
    "time_schedule": "log",
    "eps_time": 0.01,
    "use_corrector": false
  },
  "objective": 2.839988819715642
}

Associated metrics:
{
  "degree": 0.0004738899141869535,
  "wavelet": 0.001365603104578117,
  "spectre": 0.0031004946954964474,
  "clustering": 0.025257929732188437,
  "orbit": 0.0227289651712334,
  "sbm_acc": 0.734375,
  "sampling/frac_unique": 1.0,
  "sampling/frac_unique_non_iso": 1.0,
  "sampling/frac_unic_non_iso_valid": 0.078125,
  "sampling/frac_non_iso": 1.0,
  "degree_ratio": 0.7898165236449225,
  "clustering_ratio": 2.004597597792733,
  "orbit_ratio": 1.9937688746695963,
  "spectre_ratio": 2.583745579580373,
  "wavelet_ratio": 6.828015522890585,
  "average_ratio": 2.839988819715642
}

=== Best configuration ===
{
  "params": {
    "order": 30,
    "sample_target": false,
    "eps_sde": 0.001,
    "time_schedule": "log",
    "eps_time": 0.01,
    "use_corrector": false
  },
  "objective": 0.78125
}

Associated metrics:
{
  "degree": 0.00061320189994718,
  "wavelet": 0.0020734870495200397,
  "spectre": 0.002777639112707009,
  "clustering": 0.025459535038249458,
  "orbit": 0.03271249351532425,
  "sbm_acc": 0.78125,
  "sampling/frac_unique": 1.0,
  "sampling/frac_unique_non_iso": 1.0,
  "sampling/frac_unic_non_iso_valid": 0.15625,
  "sampling/frac_non_iso": 1.0,
  "degree_ratio": 1.0220031665786333,
  "clustering_ratio": 2.020598018908687,
  "orbit_ratio": 2.869516975028443,
  "spectre_ratio": 2.3146992605891743,
  "wavelet_ratio": 10.367435247600199,
  "average_ratio": 3.7188505337410276
}
"""
