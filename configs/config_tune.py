from src.parameter_search.tuning import SearchSpace

def get_tune_config(model_name, cfg):
    """
    Returns the SearchSpace, objective metric key, and metrics alias for the given model.
    """
    if model_name == "pa":
        search_space = SearchSpace(
            order=[100, 30],
            sample_target=[True],
            eps_sde=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
            eps_score=[1e-10],
            time_schedule=["log"],
            predictor=["milstein"],
            eps_time=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
            use_corrector=[False],
            snr=[2.0, 1.0, 0.1, 0.01, 0.001, 0.0001],
            scale_eps=[2.0, 1.0, 0.1, 0.01, 0.001, 0.0001],
            n_steps=[1, 2, 5],
        )
        objective = "pa_acc"
        metrics_alias = None
        
    elif model_name in ["sbm", "sbm_2comms"]:
        search_space = SearchSpace(
            order=[30],
            sample_target=[True],
            eps_sde=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
            eps_score=[1e-10],
            time_schedule=["log"],
            eps_time=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
            predictor=["heun"],
            use_corrector=[False],
            snr=[2.0, 1.0, 0.1, 0.01, 0.001, 0.0001],
            scale_eps=[2.0, 1.0, 0.1, 0.01, 0.001, 0.0001],
            n_steps=[1, 2, 5],
        )
        objective = "sbm_acc"
        metrics_alias = None
        
    elif model_name in ["tree", "tree_graphon"]:
        search_space = SearchSpace(
            num_scales=[1000],
            eps_sde=[1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10],
            time_schedule=["log"],
            eps_time=[1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10],
            predictor=["em", "heun", "milstein"],
        )
        objective = "tree_acc"
        metrics_alias = None
        
    elif model_name == "metrofi":
        search_space = SearchSpace(
            order=[cfg.sde.order] if hasattr(cfg.sde, 'order') else [30],
            sample_target=[cfg.sde.sample_target] if hasattr(cfg.sde, 'sample_target') else [True],
            eps_sde=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
            eps_score=[1e-8, 1e-9, 1e-10],
            time_schedule=[cfg.sde.time_schedule] if hasattr(cfg.sde, 'time_schedule') else ["log"],
            eps_time=[1e-2, 1e-3, 1e-4, 1e-5],
            use_corrector=[False],
            snr=[0.02, 0.01, 0.005],
            scale_eps=[1e-3, 1e-4, 1e-5],
            n_steps=[1, 2],
        )
        objective = "average_ratio"
        metrics_alias = "metrofi"
        
    else:
        raise ValueError(f"Hyperparameter tuning search space is not defined for model: {model_name}")
        
    return search_space, objective, metrics_alias
