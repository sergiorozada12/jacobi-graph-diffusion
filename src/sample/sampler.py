import math
import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.sde.sde import JacobiSDE
from src.sample.solver import PCSolver
from src.utils import quantize, adjs_to_graphs


class Sampler:
    def __init__(self, cfg, datamodule, model):
        self.cfg = cfg
        self.device = torch.device(cfg.general.device)
        self.max_num_nodes = cfg.data.max_node_num
        self.num_nodes = cfg.sampler.num_nodes

        self.model = model.to(self.device)
        self.sde = self._get_sde(self.cfg.sde)
        self.solver = self._get_solver()

        self.datamodule = datamodule
        
    def _get_sde(self, cfg_sde):
        return JacobiSDE(
            alpha=cfg_sde.alpha,
            beta=cfg_sde.beta,
            N=cfg_sde.num_scales,
            speed=cfg_sde.speed)

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
            eps=self.cfg.sampler.eps,
            device=self.device,
            order=self.cfg.sde.order,
        )

    def _make_flags(self):
        flags = torch.zeros(
            self.cfg.data.batch_size,
            self.max_num_nodes,
            dtype=torch.bool,
            device=self.device
        )
        flags[:, :self.num_nodes] = 1
        return flags

    def plot_sampled_graphs(self, graph_list, num_graphs=20):
        num_graphs = min(num_graphs, len(graph_list))
        n_cols = 5
        n_rows = math.ceil(num_graphs / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        axes = axes.flatten()

        for i in range(n_rows * n_cols):
            ax = axes[i]
            ax.axis("off")
            if i < num_graphs:
                G = graph_list[i]
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, ax=ax, node_size=50, with_labels=False)

        fig.suptitle("Sampled Graphs", fontsize=16)
        plt.tight_layout()
        
        save_path = f"samples/test.png"
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    def sample(self):
        num_rounds = math.ceil(len(self.datamodule.test_graphs) / self.cfg.data.batch_size)
        generated = []
        #for _ in range(num_rounds):
        for _ in range(1):
            flags = self._make_flags()
            adj, _ = self.solver.solve(flags)
            samples = quantize(adj)
            graphs = adjs_to_graphs(samples, is_cuda=self.device.type != 'cpu')
            generated.extend(graphs)
        self.plot_sampled_graphs(generated[:len(self.datamodule.test_graphs)])
