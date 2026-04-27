import argparse
import json
from omegaconf import OmegaConf
from pathlib import Path
import torch
import pytorch_lightning as pl

from configs.config_qm9 import MainConfig
from src.train.trainer_mol import DiffusionMolModule
from src.dataset.qm9 import QM9DatasetModule
from src.dataset.utils import DistributionNodes
from src.sample.sampler import Sampler
from src.metrics.mol_metrics import BasicMolecularMetrics

def parse_args():
    parser = argparse.ArgumentParser(description="Generate QM9 molecules.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of molecules to sample.",
    )
    parser.add_argument(
        "--vis-out",
        type=str,
        default=None,
        help="Save a molecule grid PNG to this path. Omit to skip visualization.",
    )
    parser.add_argument(
        "--vis-max",
        type=int,
        default=32,
        help="Maximum number of molecules to show in the grid.",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=None,
        help="Minimum number of nodes to sample. Requires --max-nodes. Defaults to dataset distribution.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Maximum number of nodes to sample. Requires --min-nodes. Defaults to dataset distribution.",
    )
    return parser.parse_args()

def build_node_distribution(cfg, datamodule, min_nodes=None, max_nodes=None):
    max_nodes_possible = cfg.data.max_node_num + 1
    base_prob = datamodule.node_counts() # Returns empirical distribution
    
    # Pad base_prob if it's smaller than max_nodes_possible
    if base_prob.numel() < max_nodes_possible:
        padding = torch.zeros(max_nodes_possible - base_prob.numel(), dtype=base_prob.dtype, device=base_prob.device)
        base_prob = torch.cat([base_prob, padding], dim=0)
    # Trim if it's larger
    base_prob = base_prob[:max_nodes_possible]

    if min_nodes is None and max_nodes is None:
        return DistributionNodes(prob=base_prob)

    if min_nodes is None or max_nodes is None:
        raise ValueError("Both --min-nodes and --max-nodes must be provided together.")
    if min_nodes < 1:
        raise ValueError("Minimum number of nodes must be at least 1.")
    if max_nodes < min_nodes:
        raise ValueError("Maximum number of nodes must be greater than or equal to the minimum.")

    # For OOD generation, we need to create a new probability vector
    # that goes up to max_nodes
    probs = torch.zeros(max_nodes + 1, dtype=base_prob.dtype, device=base_prob.device)
    probs[min_nodes : max_nodes + 1] = 1.0
    total_mass = probs.sum()
    if total_mass <= 0:
        raise ValueError("Probability mass over the requested range is zero.")
    probs /= total_mass
    return DistributionNodes(prob=probs)

def main():
    args = parse_args()
    cfg = OmegaConf.structured(MainConfig())
    cfg.sampler.test_graphs = args.num_samples

    if torch.cuda.is_available():
        cfg.general.device = "cuda:0"
    else:
        cfg.general.device = "cpu"

    _ = pl.seed_everything(cfg.general.seed)

    datamodule = QM9DatasetModule(cfg)
    datamodule.setup()

    if args.min_nodes is not None and args.max_nodes is not None:
        if args.max_nodes > cfg.data.max_node_num:
            cfg.data.max_node_num = args.max_nodes
            cfg.sampler.num_nodes = args.max_nodes
            print(f"Overriding max_node_num to {args.max_nodes} for OOD generation.")

    node_dist = build_node_distribution(cfg, datamodule, args.min_nodes, args.max_nodes)
    train_smiles = datamodule.train_smiles()
    sampling_metrics = BasicMolecularMetrics(dataset_info=datamodule.dataset_info, train_smiles=train_smiles)

    # We load via the module class to ensure SDEs and model are correctly linked
    module = DiffusionMolModule(
        cfg=cfg,
        sampling_metrics=sampling_metrics,
        ref_metrics=None,
        node_dist=node_dist,
    )
    
    if getattr(cfg.sde, "use_empirical_marginal", False):
        module._compute_and_set_empirical_marginals(datamodule)
    
    ckpt_dir = Path("checkpoints") / cfg.data.data
    ema_path = ckpt_dir / "weights_ema.pth"
    weights_path = ckpt_dir / "weights.pth"
    
    weight_path = ema_path if ema_path.exists() else weights_path
    if weight_path.exists():
        state_dict = torch.load(weight_path, map_location="cpu")
        module.model.load_state_dict(state_dict)
        print(f"Loaded weights from {weight_path}")
    else:
        print(f"Warning: No weights found at {ckpt_dir}. Sampling with random initialization.")

    module = module.to(cfg.general.device)
    module.eval()

    sampler = Sampler(cfg=cfg, model=module.model, node_dist=node_dist, dataset_info=datamodule.dataset_info)
    
    print(f"Sampling {args.num_samples} molecules...")
    samples, _ = sampler.sample()
    # samples is a list of PlaceHolders with [X_idx, E_idx]

    print("Computing molecular metrics...")
    sampling_metrics.reset()
    metrics = sampling_metrics.forward(
        samples,
        local_rank=0,
        test=True,
    )

    print('------------------------------------------------------------------------------------')
    print("QM9 Generation Results:")
    print(f"Validity (strict):    {metrics['validity']:.4f}")
    print(f"Uniqueness (strict):  {metrics['uniqueness']:.4f}")
    print(f"Novelty (strict):     {metrics['novelty']:.4f}")
    print("-" * 20)
    print(f"Validity (relaxed):   {metrics['relaxed_validity']:.4f}")
    print(f"Uniqueness (relaxed): {metrics['relaxed_uniqueness']:.4f}")
    print(f"Novelty (relaxed):    {metrics['relaxed_novelty']:.4f}")
    print("-" * 20)
    print(f"Mean components:      {metrics['mean_components']:.4f}")
    print(f"Mean LCC fraction:    {metrics['mean_lcc_fraction']:.4f}")
    print('------------------------------------------------------------------------------------')

    if args.vis_out:
        _visualize_molecules(samples, datamodule.dataset_info, args)

def _visualize_molecules(samples, dataset_info, args):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw, AllChem
    import os

    RDLogger.DisableLog("rdApp.*")

    atom_decoder = dataset_info.atom_decoder
    bond_dict = [
        None,
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    out_dir = Path(args.vis_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mol_list = []
    for p in samples:
        B = p.X.shape[0]
        for i in range(B):
            n = int((p.X[i] >= 0).all(dim=-1).sum().item()) if p.X[i].ndim > 1 else p.X[i].shape[0]
            # Use flag-based trimming: find first padding position (-1)
            x_i = p.X[i]   # [N] int indices
            e_i = p.E[i]   # [N, N] int indices

            # Build molecule à la DeFoG mol_from_graphs
            mol = Chem.RWMol()
            node_to_idx = {}
            for j in range(x_i.shape[0]):
                atom_idx = x_i[j].item()
                if atom_idx == -1:
                    continue
                a = Chem.Atom(atom_decoder[atom_idx])
                mol_idx = mol.AddAtom(a)
                node_to_idx[j] = mol_idx

            for r in range(e_i.shape[0]):
                for c in range(r + 1, e_i.shape[1]):
                    bond_type_idx = e_i[r, c].item()
                    if bond_type_idx < 1 or bond_type_idx >= len(bond_dict):
                        continue
                    if r in node_to_idx and c in node_to_idx:
                        mol.AddBond(node_to_idx[r], node_to_idx[c], bond_dict[bond_type_idx])

            try:
                mol = mol.GetMol()
            except Exception:
                mol = None

            if mol is not None:
                mol_list.append(mol)

    if not mol_list:
        print("No valid molecules to visualize.")
        return

    mol_list = mol_list[:args.vis_max]
    print(f"Visualizing {len(mol_list)} molecule(s) to {out_dir}/")

    # Save individual PNGs (DeFoG style)
    for i, mol in enumerate(mol_list):
        file_path = str(out_dir / f"molecule_{i}.png")
        try:
            Draw.MolToFile(mol, file_path)
        except Exception as e:
            print(f"  molecule_{i}: can't draw ({e})")

    # Also save a grid overview (DeFoG does this in visualize_chain)
    try:
        AllChem.Compute2DCoords  # check available
        for mol in mol_list:
            AllChem.Compute2DCoords(mol)
        img = Draw.MolsToGridImage(
            mol_list,
            molsPerRow=min(8, len(mol_list)),
            subImgSize=(300, 300),
        )
        grid_path = str(out_dir / "grid.png")
        img.save(grid_path)
        print(f"Grid saved to {grid_path}")
    except Exception as e:
        print(f"Could not save grid: {e}")


if __name__ == "__main__":
    main()
