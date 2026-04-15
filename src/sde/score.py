import torch
import numpy as np

from src.features.extra_features import ExtraFeatures
import torch.nn.functional as F
from src.utils import assert_symmetric_and_masked, assert_symmetric_and_masked_E, PlaceHolder
from src.features.extra_molecular_features import ExtraMolecularFeatures
from src.sde.sde import StickBreakingJacobiSDE


class JacobiScore:
    def __init__(
            self, 
            model, 
            extra_features, 
            rrwp_steps, 
            max_n_nodes, 
            order=10, 
            eps_score=1e-10, 
            sample_target=True,
            use_sampled_features=True,
            alpha=1.0,
            beta=1.0,
            direct_model_score=False,
            dataset_info=None,
            score_mode="direct_score",
        ):
        self.order = order
        self.eps = eps_score
        self.model = model
        self.score_mode = score_mode
        self.dataset_info = dataset_info
        self.feature_extractor = ExtraFeatures(
            extra_features_type=extra_features,
            rrwp_steps=rrwp_steps,
            max_n_nodes=max_n_nodes,
        )
        self.molecular_features = ExtraMolecularFeatures(dataset_info=dataset_info)
        self.sample_target = sample_target
        self.decay_cutoff = 1e-12
        self.use_sampled_features = use_sampled_features
        self.direct_model_score = direct_model_score
        if self.model is not None:
            self.model.eval()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.jacobi_a = self.beta - 1.0
        self.jacobi_b = self.alpha - 1.0

    def _jacobi_polynomials(self, x, order, a, b):
        if order <= 0:
            raise ValueError("Order must be positive")

        device, dtype = x.device, x.dtype
        a_t = torch.as_tensor(a, dtype=dtype, device=device)
        b_t = torch.as_tensor(b, dtype=dtype, device=device)
        ab = a_t + b_t
        diff = a_t - b_t
        a_sq_minus_b_sq = diff * (a_t + b_t)

        polys = []
        p_prev = torch.ones_like(x)
        polys.append(p_prev)

        if order == 1:
            return torch.stack(polys, dim=-1)

        p_curr = 0.5 * ((2.0 + ab) * x + diff)
        polys.append(p_curr)

        for n in range(1, order - 1):
            n_t = torch.as_tensor(float(n), dtype=dtype, device=device)
            two_n = 2.0 * n_t
            ab_n = two_n + ab
            ab_np1 = ab_n + 1.0
            ab_np2 = ab_np1 + 1.0

            numerator1 = ab_np1 * (ab_n * ab_np2 * x + a_sq_minus_b_sq)
            numerator2 = 2.0 * (n_t + a_t) * (n_t + b_t) * ab_np2
            denominator = 2.0 * (n_t + 1.0) * (n_t + ab + 1.0) * ab_n

            p_next = (numerator1 * p_curr - numerator2 * p_prev) / denominator
            polys.append(p_next)
            p_prev, p_curr = p_curr, p_next

        return torch.stack(polys, dim=-1)

    def jacobi_poly_and_derivative(self, x):
        P_stack = self._jacobi_polynomials(x, self.order, self.jacobi_a, self.jacobi_b)

        if self.order == 1:
            dP_stack = torch.zeros_like(P_stack)
            return P_stack, dP_stack

        shifted = self._jacobi_polynomials(
            x,
            self.order - 1,
            self.jacobi_a + 1.0,
            self.jacobi_b + 1.0,
        )

        device, dtype = x.device, x.dtype
        n = torch.arange(1, self.order, dtype=dtype, device=device)
        factors = 0.5 * (n + self.jacobi_a + self.jacobi_b + 1.0)
        shape = (1,) * x.ndim + (self.order - 1,)
        factors = factors.view(shape)

        dP_stack = torch.zeros_like(P_stack)
        dP_stack[..., 1:] = shifted * factors
        return P_stack, dP_stack

    def jacobi_score(self, adj_0, adj, t):
        orig_dtype = adj.dtype
        adj_0 = adj_0.to(torch.float64)
        adj = adj.to(torch.float64)
        t = t.to(torch.float64)

        x0 = 2.0 * adj_0 - 1.0  # [B, N, N]
        xt = 2.0 * adj - 1.0    # [B, N, N]

        P_x0, _ = self.jacobi_poly_and_derivative(x0)     # [B, N, N, order]
        P_xt, dP_xt = self.jacobi_poly_and_derivative(xt) # [B, N, N, order]

        device = adj.device
        n = torch.arange(self.order, dtype=torch.float64, device=device)
        lambdas = 0.5 * n * (n + self.alpha + self.beta - 1.0)
        lambdas = lambdas[None, None, None, :]

        a = torch.tensor(self.jacobi_a, dtype=torch.float64, device=device)
        b = torch.tensor(self.jacobi_b, dtype=torch.float64, device=device)
        raw_weights = 2.0 * n + a + b + 1.0

        log_weights = torch.empty_like(raw_weights)
        general_mask = ~( (n == 0) & (torch.abs(raw_weights) < 1e-12) )
        if general_mask.any():
            n_gen = n[general_mask]
            raw_gen = raw_weights[general_mask]
            log_weights[general_mask] = (
                torch.log(torch.abs(raw_gen))
                + torch.lgamma(n_gen + a + 1.0)
                + torch.lgamma(n_gen + b + 1.0)
                - torch.lgamma(n_gen + 1.0)
                - torch.lgamma(n_gen + a + b + 1.0)
            )

        if (~general_mask).any():
            log_weights[~general_mask] = torch.lgamma(a + 1.0) + torch.lgamma(b + 1.0)

        # Shape-agnostic broadcasting for t and lambdas
        # t is [B], adj is [B, ...]
        # poly_order is the last dimension of P_xt
        n_spatial = adj.ndim - 1
        t_view = t.view(-1, *( (1,)*n_spatial ), 1)
        
        log_decay = -t_view * lambdas + log_weights      # [B, ..., order]
        log_decay_max = log_decay.max(dim=-1, keepdim=True).values       # [B, ..., 1]
        decay_scaled = torch.exp(log_decay - log_decay_max)              # [B, ..., order]
        
        weighted_density = (decay_scaled * P_xt * P_x0).sum(dim=-1)      # [B, ...]
        weighted_grad = (decay_scaled * dP_xt * P_x0).sum(dim=-1)        # [B, ...]
        scale = torch.exp(log_decay_max.squeeze(-1))                     # [B, ...]

        density = weighted_density * scale                               # [B,N,N]
        grad_xt = 2.0 * weighted_grad * scale                            # [B,N,N]

        score = grad_xt / density.clamp_min(self.eps)
        return score.to(orig_dtype)

    def compute_score(self, state, flags, t):
        """
        Unified entry point for score computation. 
        Supports:
          - state: Tensor [B, N, N] (Legacy structural mode)
          - state: PlaceHolder (Joint node/edge mode)
        """
        # 1. State extraction and joint handling
        if isinstance(state, PlaceHolder):
            A_t = state.E
            X_t = state.X
            is_placeholder = True
        elif torch.is_tensor(state):
            # If we are in joint mode but received a tensor, we might be missing context
            # However, for molecular mode, we should really have a PlaceHolder.
            # Fix: If we have a model with attached SDEs, it's likely joint mode.
            A_t = state
            X_t = None
            is_placeholder = False
        else:
            A_t = state.E
            X_t = state.X
            is_placeholder = True

        # 2. Masking
        flags_mask = (flags[:, :, None] * flags[:, None, :]).float()
        # Handle 4D A_t (like Stick-Breaking) which needs an unsqueeze
        if A_t.ndim == 4:
             A_t = A_t * flags_mask.unsqueeze(-1)
        else:
             A_t = A_t * flags_mask
        # 3. Map v-space [0,1]^{K-1} to probability space [0,1]^K for feature extraction
        # ExtraFeatures/DeFoG logic expects probabilities (or sums of them)
        A_t_probs = A_t
        X_t_probs = X_t
        if hasattr(self.model, "sde_E") and getattr(self.model, "sde_E") is not None:
            A_t_probs = self.model.sde_E.v_to_x(A_t)
        if hasattr(self.model, "sde_X") and getattr(self.model, "sde_X") is not None:
            X_t_probs = self.model.sde_X.v_to_x(X_t) if X_t is not None else None

        # 4. Optionally sample features from distributions (DeFoG trick)
        A_t_input = self._handle_sampling(A_t_probs, is_edge=True)
        X_t_input = self._handle_sampling(X_t_probs, is_edge=False) if X_t_probs is not None else None

        # 5. Extract Extra Features
        # Structural features always come from edges/bonds (A_t_input)
        extra_structural = self.feature_extractor(A_t_input, flags)
        
        # Prepare basic X input for features and model
        X_model_input = X_t if X_t is not None else torch.zeros((*A_t.shape[:2], 0)).type_as(A_t)
        if X_model_input.ndim == 2:
            X_model_input = X_model_input.unsqueeze(-1)

        # Molecular features always come from atoms (X_model_input)
        if self.dataset_info is not None:
            noisy_data = {"X_t": X_model_input, "E_t": A_t_input, "y_t": torch.zeros(A_t.shape[0], 0).type_as(A_t)}
            extra_molecular = self.molecular_features(noisy_data)
        else:
            # Provide zero-channel placeholders for SBMs/PA to skip molecular features
            extra_molecular = PlaceHolder(
                X=torch.zeros((*X_model_input.shape[:2], 0)).type_as(X_model_input),
                E=torch.zeros((*A_t.shape[:3], 0)).type_as(A_t),
                y=torch.zeros(A_t.shape[0], 0).type_as(A_t)
            )

        # 6. Model Forward Pass
        # Pass v-space variables (A_t, X_t) as primary inputs to the model
        # but augmented with features derived from probabilities.
        # Preparation of model inputs (X, E, y)
        # Determine if noisy A_t/X_t should be concatenated based on model's expected input dimensions
        E_extra_dim = extra_structural.E.shape[-1]
        E_target_dim = self.model.input_dims["E"]
        A_t_model = A_t.unsqueeze(-1) if A_t.ndim == 3 else A_t
        if A_t_model.shape[-1] + E_extra_dim > E_target_dim:
            E_model = extra_structural.E.float()
        else:
            E_model = torch.cat([A_t_model, extra_structural.E.float()], dim=-1)

        X_extra_dim = extra_structural.X.shape[-1] + extra_molecular.X.shape[-1]
        X_target_dim = self.model.input_dims["X"]
        if X_model_input.shape[-1] + X_extra_dim > X_target_dim:
            X_model = torch.cat([extra_structural.X.float(), extra_molecular.X.float()], dim=-1)
        else:
            X_model = torch.cat([X_model_input, extra_structural.X.float(), extra_molecular.X.float()], dim=-1)
        
        # y: includes structural, molecular and time
        y_model = torch.cat([extra_structural.y.float(), extra_molecular.y.float(), t.unsqueeze(1)], dim=1).float()
        
        pred = self.model(X_model, E_model, y_model, flags)

        if self.score_mode == "graph":
            # Data Prediction Mode: pred.X and pred.E are logits for clean signal (X0, E0)
            
            # 1. Edge Score Matching
            is_sb_E = hasattr(self.model, "sde_E") and getattr(self.model, "sde_E") is not None
            if is_sb_E:
                score_E = self._compute_stick_breaking_score(pred.E, A_t, t, self.model.sde_E, flags_mask)
            else:
                score_E = self._compute_regular_score(pred.E, A_t, t, flags_mask, is_edge=True)
            
            score_E = score_E * flags_mask.unsqueeze(-1) if score_E.ndim == 4 else score_E * flags_mask
            
            # 2. Node Score Matching
            score_X = None
            if is_placeholder and X_t is not None:
                is_sb_X = hasattr(self.model, "sde_X") and getattr(self.model, "sde_X") is not None
                if is_sb_X:
                    score_X = self._compute_stick_breaking_score(pred.X, X_t, t, self.model.sde_X, None)
                else:
                    score_X = self._compute_regular_score(pred.X, X_t, t, None, is_edge=False)
            
            if is_placeholder:
                return PlaceHolder(X=score_X, E=score_E, y=None)
            return score_E

        if self.direct_model_score:
            return self._handle_direct_score(pred, flags_mask, is_placeholder)

        # 4. Score Computation for Edges (E) - Direct Score Matching mode
        is_sb_E = hasattr(self.model, "sde_E") and getattr(self.model, "sde_E") is not None
        if is_sb_E:
            score_E = self._compute_stick_breaking_score(pred.E, A_t, t, self.model.sde_E, flags_mask)
        else:
            score_E = self._compute_regular_score(pred.E, A_t, t, flags_mask, is_edge=True)

        # 5. Score Computation for Nodes (X)
        score_X = None
        if is_placeholder and X_t is not None:
            is_sb_X = hasattr(self.model, "sde_X") and getattr(self.model, "sde_X") is not None
            if is_sb_X:
                score_X = self._compute_stick_breaking_score(pred.X, X_t, t, self.model.sde_X, None)
            else:
                score_X = self._compute_regular_score(pred.X, X_t, t, None, is_edge=False)

        # 6. Final Pack
        if is_placeholder:
            return PlaceHolder(X=score_X, E=score_E, y=None)
        return score_E

    def _compute_regular_score(self, pred, state_t, t, mask, is_edge=True):
        """Standard Jacobi score for scalar variables on [0,1]"""
        if is_edge:
            clean_target = self._get_edge_target(pred, mask)
        else:
            clean_target = F.softmax(pred, dim=-1)[..., 1:].sum(dim=-1) # Fallback for scalar-like features
            
        score = self.jacobi_score(clean_target, state_t, t).float()
        if mask is not None:
            score = score * mask
        return score

    def _compute_stick_breaking_score(self, pred, state_t, t, sde_sb, mask):
        """Simplex score using stick-breaking construction"""
        X_0_dist = F.softmax(pred, dim=-1)
        
        if self.sample_target:
            # Sample discrete state from predicted distribution
            X_0_idx = torch.multinomial(X_0_dist.reshape(-1, X_0_dist.shape[-1]), 1).reshape(X_0_dist.shape[:-1])
            X_0 = F.one_hot(X_0_idx, num_classes=X_0_dist.shape[-1]).float()
        else:
            X_0 = X_0_dist
        
        v_0 = sde_sb.x_to_v(X_0)
        v_t = sde_sb.x_to_v(state_t)
        
        score_v = self.jacobi_score(v_0, v_t, t).float()
        if mask is not None:
             # v has shape [..., K-1], mask needs to broadcast
             score_v = score_v * mask.unsqueeze(-1) if mask.ndim < score_v.ndim else score_v * mask
        return score_v

    def _handle_sampling(self, val, is_edge=True):
        if not self.use_sampled_features:
            return val
        
        if is_edge:
            # Bernoulli sampling for existence of edges
            # For 3D [B,N,N] or 4D [B,N,N,C], the spatial dimensions are 1 and 2
            val_triu = torch.triu(val, diagonal=1)
            val_triu_sample = torch.bernoulli(val_triu)
            return val_triu_sample + val_triu_sample.transpose(1, 2)
        else:
            # For nodes/features on simplex, we can sample categorical
            # but usually we just use the continuous state during training.
            # If needed, we could sample one-hot. For now, matching DeFoG's feature handling:
            return val

    def _get_edge_channels_for_model(self, A_t):
        """[B, N, N] -> [B, N, N, 2] compatible with graph transformers"""
        return torch.cat([(1 - A_t).unsqueeze(-1), A_t.unsqueeze(-1)], dim=-1).float()

    def _get_edge_target(self, pred_E, mask):
        """Extract clean edge target from model prediction"""
        # Collapse multi-bond predictions to existence probability for standard Jacobi score
        # if the SDE is scalar.
        A_0_dist = F.softmax(pred_E, dim=-1)[..., 1:].sum(dim=-1).float()
        A_0_dist = A_0_dist * mask
        if self.sample_target:
            A_0_triu = torch.bernoulli(torch.triu(A_0_dist, diagonal=1))
        else:
            A_0_triu = torch.triu(A_0_dist, diagonal=1)
        return A_0_triu + A_0_triu.transpose(-1, -2)

    def _handle_direct_score(self, pred, mask, is_placeholder):
        score_E = pred.E[..., 0]
        score_E_triu = torch.triu(score_E, diagonal=1)
        score_E = (score_E_triu + score_E_triu.transpose(-1, -2)) * mask
        
        if is_placeholder:
            return PlaceHolder(X=pred.X, E=score_E, y=None)
        return score_E
