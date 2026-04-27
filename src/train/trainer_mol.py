import torch
import torch.nn.functional as F
import wandb

from src.train.base_module import DiffusionBaseModule
from src.sde.sde import JacobiSDE, StickBreakingJacobiSDE
from src.utils import node_flags, mask_x, mask_adjs, PlaceHolder
from src.features.extra_molecular_features import ExtraMolecularFeatures

class DiffusionMolModule(DiffusionBaseModule):
    def __init__(self, cfg, sampling_metrics, ref_metrics, node_dist, dataset_info=None):
        from src.metrics.train import TrainLoss
        train_loss = TrainLoss(
            lambda_node=getattr(cfg.train, "lambda_node", 1.0),
            lambda_edge=getattr(cfg.train, "lambda_edge", 1.0)
        )
        super().__init__(
            cfg,
            sampling_metrics,
            ref_metrics,
            node_dist,
            train_loss=train_loss,
            dataset_info=dataset_info,
        )
        self.molecular_features = ExtraMolecularFeatures(dataset_info=self.dataset_info)
        self.sde_X = StickBreakingJacobiSDE(
            K=cfg.dataset.node_n_types,
            alpha=cfg.sde.alpha,
            beta=cfg.sde.beta,
            s_min=cfg.sde.s_min,
            s_max=cfg.sde.s_max,
            num_scales=cfg.sde.num_scales,
        ).to(self.device)
        self.sde_E = StickBreakingJacobiSDE(
            K=cfg.dataset.edge_n_types,
            alpha=cfg.sde.alpha,
            beta=cfg.sde.beta,
            s_min=cfg.sde.s_min,
            s_max=cfg.sde.s_max,
            num_scales=cfg.sde.num_scales,
        ).to(self.device)
        self.model.sde_X = self.sde_X
        self.model.sde_E = self.sde_E
        
        if getattr(self, "use_ema", False) and getattr(self, "ema_model", None) is not None:
            self.ema_model.sde_X = self.sde_X
            self.ema_model.sde_E = self.sde_E
        
        # Re-initialize sampler so it picks up the StickBreaking SDEs from the model
        from src.sample.sampler import Sampler
        self.sampler = Sampler(cfg=cfg, model=self.model, node_dist=node_dist, dataset_info=dataset_info)
        
        self._jacobi_helper = self._build_jacobi_helper(cfg.sde)

    @staticmethod
    def _build_jacobi_helper(cfg_sde):
        from src.sde.score import JacobiScore
        helper = JacobiScore.__new__(JacobiScore)
        helper.order = cfg_sde.order
        helper.eps = cfg_sde.eps_score
        helper.alpha = cfg_sde.alpha
        helper.beta = cfg_sde.beta
        helper.jacobi_a = helper.beta - 1.0
        helper.jacobi_b = helper.alpha - 1.0 if not torch.is_tensor(helper.alpha) else helper.alpha - 1.0
        return helper

    def on_fit_start(self):
        super().on_fit_start()
        if getattr(self.cfg.sde, "use_empirical_marginal", False):
            self._compute_and_set_empirical_marginals()

    def _compute_and_set_empirical_marginals(self, datamodule=None):
        if datamodule is None:
            datamodule = self.trainer.datamodule
        dataloader = datamodule.train_dataloader()
        
        node_counts = torch.zeros(self.cfg.dataset.node_n_types, device=self.device)
        edge_counts = torch.zeros(self.cfg.dataset.edge_n_types, device=self.device)
        
        for batch in dataloader:
            if len(batch) >= 3:
                X, E, mask = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            else:
                continue
                
            node_mask = mask.bool().flatten()
            X_flat = X.reshape(-1, X.shape[-1])[node_mask]
            node_counts += X_flat.sum(dim=0)
            
            edge_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).bool()
            diag = torch.eye(mask.shape[1], device=self.device).bool()
            edge_mask = edge_mask & ~diag.unsqueeze(0)
            edge_mask_flat = edge_mask.flatten()
            
            if E.ndim == 3:
                E_one_hot = F.one_hot(E.long(), num_classes=self.cfg.dataset.edge_n_types).float()
                E_flat = E_one_hot.reshape(-1, self.cfg.dataset.edge_n_types)[edge_mask_flat]
            else:
                E_flat = E.reshape(-1, E.shape[-1])[edge_mask_flat]
                
            edge_counts += E_flat.sum(dim=0)

        node_probs = node_counts / node_counts.sum().clamp_min(1e-6)
        edge_probs = edge_counts / edge_counts.sum().clamp_min(1e-6)
        
        def get_alpha_beta(probs):
            K = len(probs)
            alpha = torch.zeros(K - 1, device=self.device)
            beta = torch.zeros(K - 1, device=self.device)
            c = 2.0 
            for i in range(K - 1):
                rem = probs[i:].sum()
                if rem > 1e-6:
                    pi = probs[i] / rem
                else:
                    pi = torch.tensor(1.0, device=self.device)
                alpha[i] = c * pi
                beta[i] = c * (1.0 - pi)
            return alpha, beta

        alpha_X, beta_X = get_alpha_beta(node_probs)
        alpha_E, beta_E = get_alpha_beta(edge_probs)
        
        self.sde_X.base_sde.alpha = alpha_X
        self.sde_X.base_sde.beta = beta_X
        self.sde_E.base_sde.alpha = alpha_E
        self.sde_E.base_sde.beta = beta_E
        
        # Update Jacobi helper (used analytically)
        self._jacobi_helper.alpha = alpha_X
        self._jacobi_helper.beta = beta_X
        self._jacobi_helper.jacobi_a = beta_X - 1.0
        self._jacobi_helper.jacobi_b = alpha_X - 1.0
        
        if getattr(self, "ema_model", None) is not None:
            self.ema_model.sde_X.base_sde.alpha = alpha_X
            self.ema_model.sde_X.base_sde.beta = beta_X
            self.ema_model.sde_E.base_sde.alpha = alpha_E
            self.ema_model.sde_E.base_sde.beta = beta_E
            
        print(f"Empirical Marginals Set! Node Alpha: {alpha_X.cpu().numpy()}, Beta: {beta_X.cpu().numpy()}")
        print(f"Empirical Marginals Set! Edge Alpha: {alpha_E.cpu().numpy()}, Beta: {beta_E.cpu().numpy()}")

    def _analytic_jacobi_score(self, v_clean, v_noisy, t):
        # Jacobi score is applied per-dimension in v-space
        return self._jacobi_helper.jacobi_score(v_clean, v_noisy, t).float()

    def _training_step_impl(self, batch_idx, X, adj, observed_mask):
        B = X.shape[0]
        flags = node_flags(adj, observed_mask)
        t = self._sample_time(B)
        
        # 1. Prepare categorical data (One-hot probabilities)
        X0 = X.float() # [B, N, 5]
        if adj.ndim == 3: # Indices -> One-hot
            E0 = F.one_hot(adj.long(), num_classes=self.cfg.dataset.edge_n_types).float()
        else:
            E0 = adj.float() # [B, N, N, 5]

        # 2. Perturb in v-space using stick-breaking
        Xt, Xt_v = self._perturb_data(X0, flags, t, sde=self.sde_X, sample=True)
        Et, Et_v = self._perturb_data(E0, flags, t, sde=self.sde_E, sample=True)
        
        # Map v-space variables back to probabilities for feature extraction
        Xt_probs = self.sde_X.v_to_x(Xt_v)
        Et_probs = self.sde_E.v_to_x(Et_v)
        
        # 3. Extract extra features (Aligned with DeFoG schema)
        # Structural: RRWP + Cycles (calculated from probabilities)
        extra_structural = self.feature_extractor(Et_probs, flags) 
        
        # Molecular: Charge, Valency, Weight (calculated from probabilities)
        noisy_data = {"X_t": Xt_probs, "E_t": Et_probs, "y_t": torch.zeros(B, 0).type_as(Xt)}
        extra_molecular = self.molecular_features(noisy_data)
        
        # 4. Concatenate for model input (X:17, E:16, y:7)
        # Primary input is v-space + features
        X_model = torch.cat([Xt_v, extra_structural.X.float(), extra_molecular.X.float()], dim=-1) # 3 + 12 + 2 = 17
        E_model = torch.cat([Et_v, extra_structural.E.float()], dim=-1)                      # 4 + 12 = 16
        y_model = torch.cat([extra_structural.y.float(), extra_molecular.y.float(), t.unsqueeze(1)], dim=1).float() # 5 + 1 + 1 = 7
        
        # 5. Run model to predict clean categories X0, E0
        # Model output dimensions are now (K_X, K_E)
        pred = self.model(X_model, E_model, y_model, flags)
        
        # 6. Loss: Cross Entropy on clean categories
        loss = self.train_loss(
            masked_pred_X=pred.X,
            true_X=X0,
            masked_pred_E=pred.E,
            true_E=E0,
            flags=flags
        )
        
        return {"loss": loss}

    def _validation_step_impl(self, X, adj, observed_mask):
        B = X.shape[0]
        flags = node_flags(adj, observed_mask)
        t = self._sample_time(B)

        X0 = X.float()
        if adj.ndim == 3:
            E0 = F.one_hot(adj.long(), num_classes=self.cfg.dataset.edge_n_types).float()
        else:
            E0 = adj.float()

        with torch.no_grad():
            Xt, Xt_v = self._perturb_data(X0, flags, t, sde=self.sde_X, sample=True)
            Et, Et_v = self._perturb_data(E0, flags, t, sde=self.sde_E, sample=True)

            Xt_probs = self.sde_X.v_to_x(Xt_v)
            Et_probs = self.sde_E.v_to_x(Et_v)

            extra_structural = self.feature_extractor(Et_probs, flags)
            noisy_data = {"X_t": Xt_probs, "E_t": Et_probs, "y_t": torch.zeros(B, 0).type_as(X)}
            extra_molecular = self.molecular_features(noisy_data)

            X_model = torch.cat([Xt_v, extra_structural.X.float(), extra_molecular.X.float()], dim=-1) # 3 + 12 + 2 = 17
            E_model = torch.cat([Et_v, extra_structural.E.float()], dim=-1)
            y_model = torch.cat([extra_structural.y.float(), extra_molecular.y.float(), t.unsqueeze(1)], dim=1).float()

            eval_model = self._get_eval_model()
            pred = eval_model(X_model, E_model, y_model, flags)

            # Compute CE losses without accumulating into TrainLoss state
            true_E_flat = E0.reshape(-1, E0.size(-1)).long()
            pred_E_flat = pred.E.reshape(-1, pred.E.size(-1))
            mask_E = (flags[:, :, None] * flags[:, None, :]).reshape(-1).bool()
            diag = torch.eye(flags.size(1), device=flags.device).unsqueeze(0).expand(B, -1, -1).reshape(-1).bool()
            mask_E = mask_E & ~diag

            true_X_flat = X0.reshape(-1, X0.size(-1)).long()
            pred_X_flat = pred.X.reshape(-1, pred.X.size(-1))
            mask_X = flags.flatten().bool()

            loss_E = F.cross_entropy(
                pred_E_flat[mask_E], true_E_flat[mask_E].argmax(dim=-1)
            ) if mask_E.any() else pred_E_flat.new_zeros(())
            loss_X = F.cross_entropy(
                pred_X_flat[mask_X], true_X_flat[mask_X].argmax(dim=-1)
            ) if mask_X.any() else pred_X_flat.new_zeros(())

            total = self.cfg.train.lambda_node * loss_X + self.cfg.train.lambda_edge * loss_E

        self.log("val/loss", total)
        self.log("val/loss_X_CE", loss_X)
        self.log("val/loss_E_CE", loss_E)
        if wandb.run:
            wandb.log({
                "val/loss": total.item(),
                "val/loss_X_CE": loss_X.item(),
                "val/loss_E_CE": loss_E.item(),
            }, commit=False)

    def _on_val_samples(self, samples):
        """Build a molecule grid from sampled PlaceHolders and log it to wandb."""
        if not wandb.run:
            return
        if self.dataset_info is None:
            return

        from rdkit import Chem, RDLogger
        from rdkit.Chem import Draw, AllChem
        import io
        from PIL import Image as PILImage

        RDLogger.DisableLog("rdApp.*")

        atom_decoder = self.dataset_info.atom_decoder
        bond_dict = [
            None,
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]

        mol_list = []
        for p in samples:
            if len(mol_list) >= 32:
                break
            B = p.X.shape[0]
            for i in range(B):
                if len(mol_list) >= 32:
                    break
                x_i = p.X[i]  # [N] — integer atom class indices
                e_i = p.E[i]  # [N, N] — integer bond class indices

                rw = Chem.RWMol()
                node_to_idx = {}
                for j in range(x_i.shape[0]):
                    atom_idx = x_i[j].item()
                    if atom_idx == -1 or atom_idx not in atom_decoder:
                        continue
                    mol_idx = rw.AddAtom(Chem.Atom(atom_decoder[atom_idx]))
                    node_to_idx[j] = mol_idx

                for r in range(e_i.shape[0]):
                    for c in range(r + 1, e_i.shape[1]):
                        b = e_i[r, c].item()
                        if b < 1 or b >= len(bond_dict):
                            continue
                        if r in node_to_idx and c in node_to_idx:
                            rw.AddBond(node_to_idx[r], node_to_idx[c], bond_dict[b])

                try:
                    mol = rw.GetMol()
                    AllChem.Compute2DCoords(mol)
                    mol_list.append(mol)
                except Exception:
                    pass

        if not mol_list:
            return

        try:
            n_cols = min(8, len(mol_list))
            img = Draw.MolsToGridImage(
                mol_list,
                molsPerRow=n_cols,
                subImgSize=(250, 250),
            )
            wandb.log({"val/molecules": wandb.Image(img)}, commit=False)
        except Exception as e:
            print(f"Warning: could not log molecule grid: {e}")
