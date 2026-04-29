import math
import torch
import numpy as np
from tqdm import trange

from src.sde.score import JacobiScore
from src.sde.sde import StickBreakingJacobiSDE
from src.utils import mask_adjs, mask_x, gen_noise, assert_symmetric_and_masked, build_time_schedule, PlaceHolder, mask_and_clamp
from src.visualization.plots import (
    plot_graph_snapshots,
    plot_heatmap_snapshots,
    save_figure,
)


class JointReverseSDE:
    """
    Wrapper to compute reverse drifts and diffusions for joint states (X and E)
    using a single pass of the score function.
    """
    def __init__(self, sde_X, sde_E, score_fn):
        self.sde_X = sde_X
        self.sde_E = sde_E
        self.score_fn = score_fn
        self.rev_X = sde_X.reverse(score_fn) if sde_X is not None else None
        self.rev_E = sde_E.reverse(score_fn)
        # Expose parameters for solvers that need them (e.g. Milstein/EM)
        self.alpha = sde_E.alpha
        self.beta = sde_E.beta

    def sde(self, state, flags, t):
        # 1. Compute joint score once
        score = self.score_fn.compute_score(state, flags, t)
        
        # 2. Extract drifts and diffusions using the precomputed score
        drift_E, diff_E, _ = self.rev_E.drift_from_score(state.E, flags, t, score.E)
        
        drift_X, diff_X = None, None
        if state.X is not None and self.rev_X is not None:
             drift_X, diff_X, _ = self.rev_X.drift_from_score(state.X, flags, t, score.X)
        
        return PlaceHolder(X=drift_X, E=drift_E, y=None), PlaceHolder(X=diff_X, E=diff_E, y=None)

    def sde_with_diffusion_grad(self, state, flags, t):
        # 1. Compute joint score once
        score = self.score_fn.compute_score(state, flags, t)
        
        # 2. Extract drifts, diffusions, and grads using the precomputed score
        drift_E, diff_E, grad_E = self.rev_E.drift_from_score(state.E, flags, t, score.E)
        
        drift_X, diff_X, grad_X = None, None, None
        if state.X is not None and self.rev_X is not None:
             drift_X, diff_X, grad_X = self.rev_X.drift_from_score(state.X, flags, t, score.X)
        
        return (PlaceHolder(X=drift_X, E=drift_E, y=None), 
                PlaceHolder(X=diff_X, E=diff_E, y=None), 
                PlaceHolder(X=grad_X, E=grad_E, y=None))


class EulerMaruyamaPredictor:
    def __init__(self, sde, score_fn, rsde=None):
        self.sde = sde
        self.rsde = rsde if rsde is not None else sde.reverse(score_fn)
        self.score_fn = score_fn

    @torch.no_grad()
    def update(self, state, flags, t, dt):
        """
        state: Tensor [B, N, N] or PlaceHolder
        """
        is_placeholder = isinstance(state, PlaceHolder)
        device = state.E.device if is_placeholder else state.device
        dtype = state.E.dtype if is_placeholder else state.dtype
        
        # 1. Reverse SDE Step
        # rsde.sde returns drifts/diffusions matching the state structure
        drift, diffusion = self.rsde.sde(state, flags, t)
        
        # 2. Diffusion step
        sqrt_mdt = torch.sqrt(torch.tensor(-dt, device=device, dtype=dtype))
        noise = gen_noise(state, flags).to(device=device, dtype=dtype)
        
        if not is_placeholder:
            state_mean = state + drift * dt
            state_new  = state_mean + diffusion * sqrt_mdt * noise
        else:
            # Joint state (PlaceHolder)
            E_mean = state.E + drift.E * dt
            E_new  = E_mean + diffusion.E * sqrt_mdt * noise.E
            
            X_mean, X_new = None, None
            if state.X is not None:
                X_mean = state.X + drift.X * dt
                X_new  = X_mean + diffusion.X * sqrt_mdt * noise.X
            
            state_new = PlaceHolder(X=X_new, E=E_new, y=state.y)
            state_mean = PlaceHolder(X=X_mean, E=E_mean, y=state.y)

        # Unified masking and clamping
        state_new = mask_and_clamp(state_new, flags)
        state_mean = mask_and_clamp(state_mean, flags)

        return state_new, state_mean

        return state_new, state_mean

class HeunPredictor:
    def __init__(self, sde, score_fn, rsde=None):
        self.sde = sde
        self.rsde = rsde if rsde is not None else sde.reverse(score_fn)
        self.score_fn = score_fn

    @torch.no_grad()
    def update(self, state, flags, t, dt):
        """
        Heun's method (Trapezoidal rule) for SDEs.
        state: Tensor [B, N, N] or PlaceHolder
        """
        is_placeholder = isinstance(state, PlaceHolder)
        device = state.E.device if is_placeholder else state.device
        dtype = state.E.dtype if is_placeholder else state.dtype

        dt_tensor = torch.tensor(dt, device=device, dtype=dtype)
        sqrt_mdt = torch.sqrt(torch.tensor(-dt, device=device, dtype=dtype))
        noise = gen_noise(state, flags).to(device=device, dtype=dtype)
        
        # 1. First stage (Euler step)
        drift1, diff1 = self.rsde.sde(state, flags, t)
        
        if not is_placeholder:
            dW = sqrt_mdt * noise
            state_tilde = state + drift1 * dt_tensor + diff1 * dW
        else:
            E_tilde = state.E + drift1.E * dt_tensor + diff1.E * (sqrt_mdt * noise.E)
            X_tilde = None
            if state.X is not None:
                X_tilde = state.X + drift1.X * dt_tensor + diff1.X * (sqrt_mdt * noise.X)
            state_tilde = PlaceHolder(X=X_tilde, E=E_tilde, y=state.y)

        state_tilde = mask_and_clamp(state_tilde, flags)

        # 2. Second stage
        drift2, diff2 = self.rsde.sde(state_tilde, flags, t)

        if not is_placeholder:
            drift_avg = 0.5 * (drift1 + drift2)
            diff_avg = 0.5 * (diff1 + diff2)
            state_mean = state + drift_avg * dt_tensor
            state_new = state + drift_avg * dt_tensor + diff_avg * (sqrt_mdt * noise)
        else:
            driftE_avg = 0.5 * (drift1.E + drift2.E)
            diffE_avg  = 0.5 * (diff1.E + diff2.E)
            E_mean = state.E + driftE_avg * dt_tensor
            E_new  = state.E + driftE_avg * dt_tensor + diffE_avg * (sqrt_mdt * noise.E)
            
            X_new, X_mean = None, None
            if state.X is not None:
                driftX_avg = 0.5 * (drift1.X + drift2.X)
                diffX_avg  = 0.5 * (diff1.X + diff2.X)
                X_mean = state.X + driftX_avg * dt_tensor
                X_new  = state.X + driftX_avg * dt_tensor + diffX_avg * (sqrt_mdt * noise.X)
            
            state_new = PlaceHolder(X=X_new, E=E_new, y=state.y)
            state_mean = PlaceHolder(X=X_mean, E=E_mean, y=state.y)

        state_new = mask_and_clamp(state_new, flags)
        state_mean = mask_and_clamp(state_mean, flags)

        return state_new, state_mean

class MilsteinPredictor:
    def __init__(self, sde, score_fn, rsde=None):
        self.sde = sde
        self.rsde = rsde if rsde is not None else sde.reverse(score_fn)
        self.score_fn = score_fn

    @torch.no_grad()
    def update(self, state, flags, t, dt):
        """
        Milstein's method for SDEs.
        state: Tensor [B, N, N] or PlaceHolder
        """
        is_placeholder = isinstance(state, PlaceHolder)
        device = state.E.device if is_placeholder else state.device
        dtype = state.E.dtype if is_placeholder else state.dtype

        dt_tensor = torch.tensor(dt, device=device, dtype=dtype)
        sqrt_mdt = torch.sqrt(torch.tensor(-dt, device=device, dtype=dtype))
        noise = gen_noise(state, flags).to(device=device, dtype=dtype)
        
        # Get drift, diffusion, and its gradient
        # rsde should return PlaceHolders for all three if joint
        drift, diffusion, diff_grad = self.rsde.sde_with_diffusion_grad(state, flags, t)
        
        if not is_placeholder:
            dW = sqrt_mdt * noise
            state_mean = state + drift * dt_tensor
            state_new = state_mean + diffusion * dW + 0.5 * diffusion * diff_grad * (dW * dW - dt_tensor)
        else:
            dW_E = sqrt_mdt * noise.E
            E_mean = state.E + drift.E * dt_tensor
            E_new = E_mean + diffusion.E * dW_E + 0.5 * diffusion.E * diff_grad.E * (dW_E * dW_E - dt_tensor)
            
            X_new, X_mean = None, None
            if state.X is not None:
                dW_X = sqrt_mdt * noise.X
                X_mean = state.X + drift.X * dt_tensor
                X_new = X_mean + diffusion.X * dW_X + 0.5 * diffusion.X * diff_grad.X * (dW_X * dW_X - dt_tensor)
            
            state_new = PlaceHolder(X=X_new, E=E_new, y=state.y)
            state_mean = PlaceHolder(X=X_mean, E=E_mean, y=state.y)

        state_new = mask_and_clamp(state_new, flags)
        state_mean = mask_and_clamp(state_mean, flags)

        return state_new, state_mean

class LangevinCorrector:
    def __init__(self, sde, score_fn, snr, scale_eps, n_steps, eps):
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.scale_eps = scale_eps
        self.n_steps = n_steps
        self.eps = eps

    def update(self, state, flags, t):
        is_placeholder = isinstance(state, PlaceHolder)
        device = state.E.device if is_placeholder else state.device
        dtype = state.E.dtype if is_placeholder else state.dtype
        B = state.E.shape[0] if is_placeholder else state.shape[0]
        mask = flags.unsqueeze(2) * flags.unsqueeze(1) if not is_placeholder else (flags[:, :, None] * flags[:, None, :]).float()

        for _ in range(self.n_steps):
            score = self.score_fn.compute_score(state, flags, t)
            
            if not is_placeholder:
                precond = (state * (1.0 - state)).clamp_min(1e-12)
                score = precond * score
                noise = gen_noise(state, flags).to(device=device, dtype=dtype)
                
                # Step size logic
                f_norm = self._masked_norm(score, mask, B)
                n_norm = self._masked_norm(noise, mask, B)
                step_size = 2.0 * (self.snr * n_norm / f_norm).pow(2)
                
                state = state + step_size.view(B, 1, 1) * score + torch.sqrt(2.0 * step_size).view(B, 1, 1) * noise * self.scale_eps
                state = state.clamp(0.0, 1.0)
            else:
                # Joint Corrector Step
                # We apply Langevin to E and X components
                # For simplicity, we use a single step size based on E norm
                # (Can be refined in the future)
                E0, X0 = state.E, state.X
                scoreE, scoreX = score.E, score.X
                
                # E preconditioning
                precondE = (E0 * (1.0 - E0)).clamp_min(1e-12)
                scoreE = precondE * scoreE
                noiseE = gen_noise(E0, flags).to(device=device, dtype=dtype)
                
                f_norm_E = self._masked_norm(scoreE, mask, B)
                n_norm_E = self._masked_norm(noiseE, mask, B)
                step_size = 2.0 * (self.snr * n_norm_E / f_norm_E).pow(2)
                ss = step_size.view(B, 1, 1) if E0.ndim == 3 else step_size.view(B, 1, 1, 1)
                
                E_new = E0 + ss * scoreE + torch.sqrt(2.0 * step_size).view(B, 1, 1, 1 if E0.ndim > 3 else 1) * noiseE
                
                X_new = X0
                if X0 is not None:
                     precondX = (X0 * (1.0 - X0)).clamp_min(1e-12)
                     scoreX = precondX * scoreX
                     noiseX = torch.randn_like(X0)
                     X_new = X0 + step_size.view(B, 1, 1) * scoreX + torch.sqrt(2.0 * step_size).view(B, 1, 1) * noiseX
                
                state = PlaceHolder(X=X_new, E=E_new, y=state.y)
                # Re-apply masking/clamping
                state = mask_and_clamp(state, flags)

        return state, state

    def _masked_norm(self, X, mask, B):
        # Handle simplex masking
        m = mask.unsqueeze(-1) if X.ndim > mask.ndim else mask
        Xf = (X * m).reshape(B, -1)
        return Xf.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class PCSolver:
    def __init__(
            self,
            sde,
            shape_adj,
            model,
            node_features,
            rrwp_steps,
            max_n_nodes,
            snr=0.1,
            scale_eps=1.0,
            n_steps=1,
            denoise=True,
            eps=1e-3,
            device="cuda",
            order=10,
            sample_target=True,
            eps_corrector=1e-5,
            eps_score=1e-10,
            use_corrector=False,
            predictor_type="em",
            time_schedule="log",
            time_schedule_power=2.0,
            use_sampled_features=True,
            score_mode="graph",
            dataset_info=None,
        ):
        self.sde = sde
        self.shape_adj = shape_adj
        self.denoise = denoise
        self.eps = eps
        self.device = device
        self.n_steps = n_steps
        self.use_corrector = use_corrector
        self.time_schedule = time_schedule
        self.time_schedule_power = time_schedule_power
        self.score_mode = score_mode
        self.model = model
        
        jacobi_score = JacobiScore(
            model=model,
            extra_features=node_features,
            rrwp_steps=rrwp_steps,
            max_n_nodes=max_n_nodes,
            order=order,
            sample_target=sample_target,
            eps_score=eps_score,
            use_sampled_features=use_sampled_features,
            alpha=sde.alpha,
            beta=sde.beta,
            direct_model_score=(score_mode == "direct_score"),
            dataset_info=dataset_info,
            score_mode=score_mode,
        )

        predictor_type = (predictor_type or "em").lower()
        
        # For molecules, we wrap sde_X and sde_E into a JointReverseSDE
        if score_mode == "molecular":
            sde_X = getattr(model, "sde_X", None)
            sde_E = getattr(model, "sde_E", sde)
            self.rsde = JointReverseSDE(sde_X, sde_E, jacobi_score)
        else:
            self.rsde = sde.reverse(jacobi_score)

        if predictor_type == "milstein":
            self.predictor = MilsteinPredictor(sde, jacobi_score, rsde=self.rsde)
        elif predictor_type == "heun":
            self.predictor = HeunPredictor(sde, jacobi_score, rsde=self.rsde)
        else:
            self.predictor = EulerMaruyamaPredictor(sde, jacobi_score, rsde=self.rsde)
        
        self.corrector = LangevinCorrector(sde, jacobi_score, snr, scale_eps, n_steps, eps_corrector)

    def solve(self, flags):
        with torch.no_grad():
            is_joint = hasattr(self.sde, "K") or hasattr(self.sde, "base_sde") # Heuristic for Joint/SB
            
            # 1. Prior Sampling
            if not is_joint:
                state = self.sde.prior_sampling(self.shape_adj).to(self.device)
                state = mask_adjs(state, flags)
            else:
                # Joint state initialization (Molecule/Categorical style)
                # sde here is likely a wrapper or we need to access sde_X/sde_E
                E_prior = self.model.sde_E.prior_sampling(self.shape_adj).to(self.device)
                X_prior = None
                if self.model.sde_X is not None:
                     X_prior = self.model.sde_X.prior_sampling((self.shape_adj[0], self.shape_adj[1])).to(self.device)
                
                # Keep in v-space during sampling! 
                # Model and SDE logic in JGD are defined in [0,1] (v-space for categorical)
                
                state = PlaceHolder(X=X_prior, E=E_prior, y=None)
                state = mask_and_clamp(state, flags)

            history = []
            # We use sde_E as the time reference
            sde_ref = self.model.sde_E if is_joint else self.sde
            N = sde_ref.N
            ts = build_time_schedule(
                num_scales=N,
                T=sde_ref.T,
                eps=self.eps,
                kind=self.time_schedule,
                power=self.time_schedule_power,
            ).to(self.device, dtype=E_prior.dtype if is_joint else state.dtype)
            
            for i in trange(0, N, desc="[Sampling]", position=1, leave=False):
                t, dt  = ts[i], (ts[i+1] - ts[i]).item()
                vec_t  = torch.full((self.shape_adj[0],), t, device=self.device, dtype=ts.dtype)

                state, state_mean = self.predictor.update(state, flags, vec_t, dt)
                if self.use_corrector:
                    state, state_mean = self.corrector.update(state, flags, vec_t)
                
                history.append(state.E[0].detach().cpu() if is_joint else state[0].detach().cpu())
            
        flags0 = flags[0].cpu().bool()
        active_idx = flags0.nonzero(as_tuple=True)[0]

        # pick 100 evenly spaced snapshots
        idxs = np.linspace(0, len(history) - 1, 100, dtype=int)
        if not is_joint:
            graph_fig = plot_graph_snapshots(
                history,
                grid_shape=(5, 5) if len(history) >= 25 else (1, len(history)),
            )
            heatmap_fig = plot_heatmap_snapshots(
                history,
                grid_shape=(5, 5) if len(history) >= 25 else (1, len(history)),
            )
            save_figure(graph_fig, "graph_snapshots.png")
            save_figure(heatmap_fig, "history_heatmaps.png")

        return ((state_mean if self.denoise else state), N * (self.n_steps + 1))
