import math
import torch

from src.sde.sde import JacobiSDE
from src.sample.solver import PCSolver
from src.utils import quantize, adjs_to_graphs
from src.visualization.plots import plot_graph_grid, plot_weighted_adj_and_graph


class Sampler:
    def __init__(self, cfg, model, node_dist=None, dataset_info=None):
        self.cfg = cfg
        self.device = torch.device(cfg.general.device)
        self.max_num_nodes = cfg.data.max_node_num
        self.num_nodes = cfg.sampler.num_nodes
        self.score_mode = getattr(cfg.train, "training_mode", "graph")

        self.use_sampled_features = getattr(cfg.model, "use_sampled_features", True)
        self.model = model.to(self.device)
        self.dataset_info = dataset_info
        self.sde = self._get_sde(self.cfg.sde)
        self.solver = self._get_solver()

        self.node_dist = node_dist
        
    def _get_sde(self, cfg_sde):
        # We check if the model has attached SDEs (which happens in base_module)
        if hasattr(self.model, "sde_E") and self.model.sde_E is not None:
            return self.model.sde_E
        
        from src.sde.sde import JacobiSDE
        return JacobiSDE(
            alpha=cfg_sde.alpha,
            beta=cfg_sde.beta,
            num_scales=cfg_sde.num_scales,
            s_min=cfg_sde.s_min,
            s_max=cfg_sde.s_max,
            eps=cfg_sde.eps_sde,
        )

    def _get_solver(self):
        shape_adj = (self.cfg.data.batch_size, self.max_num_nodes, self.max_num_nodes)

        return PCSolver(
            sde=self.sde,
            shape_adj=shape_adj,
            model=self.model,
            node_features=self.cfg.model.extra_features_type,
            rrwp_steps=self.cfg.model.rrwp_steps,
            max_n_nodes=self.max_num_nodes,
            snr=self.cfg.sampler.snr,
            scale_eps=self.cfg.sampler.scale_eps,
            n_steps=self.cfg.sampler.n_steps,
            denoise=self.cfg.sampler.noise_removal,
            eps=self.cfg.sampler.eps_time,
            device=self.device,
            order=self.cfg.sde.order,
            sample_target=self.cfg.sde.sample_target,
            eps_score=self.cfg.sde.eps_score,
            use_corrector=self.cfg.sampler.use_corrector,
            predictor_type=getattr(self.cfg.sampler, "predictor", "em"),
            dataset_info=self.dataset_info,
            time_schedule=self.cfg.sampler.time_schedule,
            time_schedule_power=self.cfg.sampler.time_schedule_power,
            use_sampled_features=self.use_sampled_features,
            score_mode=self.score_mode,
        )

    def set_model(self, model):
        self.model = model.to(self.device)
        self.model.eval()
        self.solver.predictor.score_fn.model = self.model
        self.solver.corrector.score_fn.model = self.model

    def _make_flags(self, use_node_dist: bool = True):
        flags = torch.zeros(
            self.cfg.data.batch_size,
            self.max_num_nodes,
            dtype=torch.bool,
            device=self.device
        )

        if use_node_dist and self.node_dist:
            num_nodes = self.node_dist.sample_n(self.cfg.data.batch_size, self.device)
            for i, n in enumerate(num_nodes):
                flags[i, :n] = 1
        else:
            flags[:, :self.num_nodes] = 1
        return flags

    def plot_sampled_graphs_(self, graph_list, num_graphs=20):
        return plot_graph_grid(
            graph_list,
            num_graphs=num_graphs,
            n_cols=5,
            dataset_name=self.cfg.data.data,
            show_avg_degree=False,
        )
    
    def plot_sampled_graphs(self, graph_list, num_graphs=100):
        return plot_graph_grid(
            graph_list,
            num_graphs=min(100, num_graphs, len(graph_list)),
            n_cols=10,
            n_rows=10,
            dataset_name=self.cfg.data.data,
        )

    def sample(
        self,
        keep_isolates: bool = False,
        return_adjs: bool = False,
        *,
        use_node_dist: bool = True,
        nodelist=None,
        keep_zero_weights: bool = False,
    ):
        num_rounds = math.ceil(self.cfg.sampler.test_graphs / self.cfg.data.batch_size)
        generated = []
        from src.utils import PlaceHolder
        collected_data = []
        is_joint_flag = False
        
        for r in range(num_rounds):
            flags = self._make_flags(use_node_dist=use_node_dist)
            state, _ = self.solver.solve(flags)
            
            if not isinstance(state, PlaceHolder):
                adj = state
                if self.score_mode == "weighted":
                     samples = adj.clamp(0.0, 1.0)
                else:
                     samples = quantize(adj)
                
                graphs = adjs_to_graphs(
                    samples,
                    is_cuda=self.device.type != 'cpu',
                    keep_isolates=keep_isolates,
                    nodelist=nodelist,
                    keep_zero_weights=keep_zero_weights,
                )
                generated.extend(graphs)
                collected_data.append(samples.detach().cpu())
            else:
                is_joint_flag = True
                # Joint mode (Molecule)
                # state is a PlaceHolder with X and E (simplex)
                # Quantize: argmax for bond types and atom types
                E_idx = torch.argmax(state.E, dim=-1) # [B, N, N] contains 0..4
                X_idx = torch.argmax(state.X, dim=-1) # [B, N]
                
                samples = PlaceHolder(X=X_idx, E=E_idx, y=None)
                generated.append(samples) # We return PlaceHolders for molecules to be built by Metrics
                collected_data.append(samples)

        if not is_joint_flag:
             fig = self.plot_sampled_graphs(generated)
        else:
             # Molecules have custom plotting via metrics module usually
             fig = None 

        if return_adjs:
            return generated, fig, collected_data
        return generated, fig
