import math
from pathlib import Path
import torch

from src.sde.sde import JacobiSDE
from src.sample.solver import PCSolver
from src.utils import quantize, adjs_to_graphs
from src.visualization.plots import plot_graph_grid, plot_weighted_adj_and_graph


class Sampler:
    def __init__(self, cfg, model, node_dist=None, artifact_dir=None):
        self.cfg = cfg
        self.device = torch.device(cfg.general.device)
        self.max_num_nodes = cfg.data.max_node_num
        self.num_nodes = cfg.sampler.num_nodes
        self.score_mode = getattr(cfg.train, "training_mode", "graph")

        self.use_sampled_features = getattr(cfg.model, "use_sampled_features", True)
        self.model = model.to(self.device)
        self.sde = self._get_sde(self.cfg.sde)
        self.artifact_dir = artifact_dir
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
            time_schedule=self.cfg.sampler.time_schedule,
            time_schedule_power=self.cfg.sampler.time_schedule_power,
            use_sampled_features=self.use_sampled_features,
            score_mode=self.score_mode,
            conditional=getattr(self.cfg.model, "conditional", False),
            condition_dim=getattr(self.cfg.model, "condition_dim", 3),
            guidance_scale=getattr(self.cfg.sampler, "guidance_scale", 0.0),
            positional_encoding=getattr(self.cfg.model, "positional_encoding", False),
            positional_encoding_dim=getattr(self.cfg.model, "positional_encoding_dim", 0),
            artifact_dir=self.artifact_dir,
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
        condition=None,
        fixed_flags=None,
        resume_path=None,
    ):
        if condition is not None and fixed_flags is not None and condition.size(0) != fixed_flags.size(0):
            raise ValueError("condition and fixed_flags must contain the same number of graphs.")
        if fixed_flags is not None:
            total_samples = fixed_flags.size(0)
        elif condition is not None:
            total_samples = condition.size(0)
        else:
            total_samples = self.cfg.sampler.test_graphs
        num_rounds = math.ceil(total_samples / self.cfg.data.batch_size)
        generated = []
        first_adj = None
        first_flags = None
        collected_adjs = []
        start_round = 0
        resume_file = Path(resume_path) if resume_path is not None else None
        if resume_file is not None and resume_file.exists():
            state = torch.load(resume_file, map_location="cpu")
            if int(state["total_samples"]) != int(total_samples):
                raise ValueError("Resume state sample count does not match this evaluation.")
            saved_flags = state.get("fixed_flags")
            expected_flags = fixed_flags.detach().cpu() if fixed_flags is not None else None
            if saved_flags is not None and not torch.equal(saved_flags, expected_flags):
                raise ValueError("Resume state node masks do not match this evaluation.")
            saved_condition = state.get("condition")
            expected_condition = condition.detach().cpu() if condition is not None else None
            if saved_condition is not None and not torch.equal(saved_condition, expected_condition):
                raise ValueError("Resume state conditions do not match this evaluation.")
            restored = state["adjs"]
            collected_adjs.append(restored)
            start_round = math.ceil(restored.size(0) / self.cfg.data.batch_size)
            torch.set_rng_state(state["torch_rng_state"])
            if self.device.type == "cuda" and state.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(state["cuda_rng_state"], self.device)
            first_adj = restored[0]
            if fixed_flags is not None:
                first_flags = fixed_flags[0].detach().cpu()
            generated.extend(adjs_to_graphs(
                restored, is_cuda=False, keep_isolates=keep_isolates,
                nodelist=nodelist, keep_zero_weights=keep_zero_weights,
            ))

        for round_idx in range(start_round, num_rounds):
            start = round_idx * self.cfg.data.batch_size
            end = min(start + self.cfg.data.batch_size, total_samples)
            round_batch_size = end - start
            if fixed_flags is not None:
                flags = fixed_flags[start:end].to(self.device)
            else:
                flags = self._make_flags(use_node_dist=use_node_dist)[:round_batch_size]
            cond_batch = None
            if condition is not None:
                cond_batch = condition[start:end].to(self.device)
            adj, _ = self.solver.solve(flags, condition=cond_batch)
            if self.score_mode == "weighted":
                samples = adj.clamp(0.0, 1.0)
            else:
                samples = quantize(adj)
            collected_adjs.append(samples.detach().cpu())
            if first_adj is None:
                first_adj = samples[0].detach().cpu()
                first_flags = flags[0].detach().cpu()
            graphs = adjs_to_graphs(
                samples,
                is_cuda=self.device.type != 'cpu',
                keep_isolates=keep_isolates,
                nodelist=nodelist,
                keep_zero_weights=keep_zero_weights,
            )
            generated.extend(graphs)
            if resume_file is not None:
                resume_file.parent.mkdir(parents=True, exist_ok=True)
                state = {
                    "total_samples": total_samples,
                    "adjs": torch.cat(collected_adjs, dim=0),
                    "fixed_flags": fixed_flags.detach().cpu() if fixed_flags is not None else None,
                    "condition": condition.detach().cpu() if condition is not None else None,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
                }
                tmp_file = resume_file.with_name(resume_file.name + ".tmp")
                torch.save(state, tmp_file)
                tmp_file.replace(resume_file)
            #for graph in graphs:
            #    largest_cc = max(nx.connected_components(graph), key=len)
            #    generated.append(graph.subgraph(largest_cc).copy())
        if self.score_mode == "weighted" and first_adj is not None:
            fig = plot_weighted_adj_and_graph(
                first_adj,
                flags=first_flags,
                dataset_name=self.cfg.data.data,
            )
        else:
            fig = self.plot_sampled_graphs(generated)
        if return_adjs:
            adjs_stacked = torch.cat(collected_adjs, dim=0)
            return generated, fig, adjs_stacked
        return generated, fig
