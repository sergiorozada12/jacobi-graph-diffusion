import math
import torch

from src.sde.sde import JacobiSDE
from src.sample.solver import PCSolver
from src.utils import quantize, adjs_to_graphs
from src.visualization.plots import plot_graph_grid


class Sampler:
    def __init__(self, cfg, model, node_dist=None):
        self.cfg = cfg
        self.device = torch.device(cfg.general.device)
        self.max_num_nodes = cfg.data.max_node_num
        self.num_nodes = cfg.sampler.num_nodes
        self.score_mode = getattr(cfg.train, "training_mode", "graph")

        self.use_sampled_features = getattr(cfg.model, "use_sampled_features", True)
        self.model = model.to(self.device)
        self.sde = self._get_sde(self.cfg.sde)
        self.solver = self._get_solver()

        self.node_dist = node_dist
        
    def _get_sde(self, cfg_sde):
        return JacobiSDE(
            alpha=cfg_sde.alpha,
            beta=cfg_sde.beta,
            N=cfg_sde.num_scales,
            s_min=cfg_sde.s_min,
            s_max=cfg_sde.s_max,
            eps=cfg_sde.eps_sde,
            max_force=cfg_sde.max_force,
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
            eps_score_dist=self.cfg.sde.eps_score_dist,
            use_corrector=self.cfg.sampler.use_corrector,
            time_schedule=self.cfg.sde.time_schedule,
            time_schedule_power=self.cfg.sde.time_schedule_power,
            use_sampled_features=self.use_sampled_features,
            score_mode=self.score_mode,
        )

    def set_model(self, model):
        self.model = model.to(self.device)
        self.model.eval()
        self.solver.predictor.score_fn.model = self.model
        self.solver.corrector.score_fn.model = self.model

    def _make_flags(self):
        flags = torch.zeros(
            self.cfg.data.batch_size,
            self.max_num_nodes,
            dtype=torch.bool,
            device=self.device
        )

        if self.node_dist:
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

    def sample(self):
        num_rounds = math.ceil(self.cfg.sampler.test_graphs / self.cfg.data.batch_size)
        generated = []
        for _ in range(num_rounds):
        # for _ in range(1):
            flags = self._make_flags()
            adj, _ = self.solver.solve(flags)
            samples = quantize(adj)
            graphs = adjs_to_graphs(samples, is_cuda=self.device.type != 'cpu')
            generated.extend(graphs)
            #for graph in graphs:
            #    largest_cc = max(nx.connected_components(graph), key=len)
            #    generated.append(graph.subgraph(largest_cc).copy())
        fig = self.plot_sampled_graphs(generated)
        return generated, fig
