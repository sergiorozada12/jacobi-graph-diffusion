import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
import pickle
from types import SimpleNamespace
from torch_geometric.datasets import QM9 as PyGQM9

from src.utils import PlaceHolder, node_flags, mask_adjs, mask_x

class QM9DatasetModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.data.batch_size
        self.max_node_num = cfg.data.max_node_num
        self.remove_h = getattr(cfg.data, "remove_h", True)
        self.aromatic = getattr(cfg.data, "aromatic", True)
        
        # Atom and Bond mapping consistent with DeFoG (H removed, remove_h=True)
        self.atom_types = ["C", "N", "O", "F"]
        self.atom_decoder = {i: s for i, s in enumerate(self.atom_types)}
        self.atom_encoder = {s: i for i, s in enumerate(self.atom_types)}
        
        self.bond_types = [Chem.rdchem.BondType.SINGLE, 
                           Chem.rdchem.BondType.DOUBLE, 
                           Chem.rdchem.BondType.TRIPLE, 
                           Chem.rdchem.BondType.AROMATIC]
        self.num_atom_types = len(self.atom_types)
        self.num_bond_types = len(self.bond_types) + 1  # +1 for "No bond"
        
        self.dataset_info = SimpleNamespace(**{
            "atom_types": self.atom_types,
            "atom_decoder": self.atom_decoder,
            "atom_encoder": self.atom_encoder,
            "bond_types": self.bond_types,
            "num_atom_types": self.num_atom_types,
            "num_bond_types": self.num_bond_types,
            "max_node_num": self.max_node_num,
            "valencies": [4, 3, 2, 1],         # C, N, O, F
            "atom_weights": {0: 12, 1: 14, 2: 16, 3: 19},  # C, N, O, F
            "max_weight": 390
        })
        self._train_smiles_cache = None

    def setup(self, stage=None):
        # We expect raw data in DeFoG's directory as specified in the plan
        # or in the local data directory.
        data_dir = self.cfg.data.dir
        raw_path = os.path.join(data_dir, "qm9_raw.pkl")
        
        if not os.path.exists(raw_path):
             # 1. Try to find DeFoG's raw data
             defog_raw = "/home/srozada/DeFoG/data/qm9/qm9_pyg/raw/gdb9.sdf"
             # 2. Try to find GruM's data
             grum_csv = "/home/srozada/GruM/data/qm9.csv"
             
             if os.path.exists(defog_raw):
                 print(f"Propagating raw data from {defog_raw}...")
                 self._process_raw_data(defog_raw, raw_path)
             elif os.path.exists(grum_csv):
                 print(f"Loading data from GruM CSV: {grum_csv}...")
                 # Simplified CSV to pkl conversion for the module's existing logic
                 self._process_csv_data(grum_csv, raw_path)
             else:
                 print("Local data not found. Attempting automated PyG download...")
                 self._setup_pyg_data()
                 return # _setup_pyg_data sets train/val/test_ds directly

        with open(raw_path, "rb") as f:
            dataset = pickle.load(f)
            
        self.train_ds = self._build_dataset(dataset["train"])
        self.val_ds = self._build_dataset(dataset["val"])
        self.test_ds = self._build_dataset(dataset["test"])

    def _process_raw_data(self, sdf_path, output_path):
        # Implementation of raw processing matching DeFoG
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=self.remove_h, sanitize=True)
        mols = [m for m in tqdm(suppl, desc="Reading QM9 SDF") if m is not None]
        
        # Simple split for demonstration if not provided
        n = len(mols)
        indices = np.random.permutation(n)
        train_idx = indices[:int(0.8*n)]
        val_idx = indices[int(0.8*n):int(0.9*n)]
        test_idx = indices[int(0.9*n):]
        
        def mol_to_data(mol):
            n_nodes = mol.GetNumAtoms()
            if n_nodes > self.max_node_num: return None
            
            x = torch.zeros(self.max_node_num, dtype=torch.long)
            for i, atom in enumerate(mol.GetAtoms()):
                x[i] = self.atom_encoder[atom.GetSymbol()]
            
            adj = torch.zeros(self.max_node_num, self.max_node_num, dtype=torch.long)
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bt = bond.GetBondType()
                for k, type in enumerate(self.bond_types):
                    if bt == type:
                        adj[i, j] = adj[j, i] = k + 1
                        break
            
            # Mask
            mask = torch.zeros(self.max_node_num)
            mask[:n_nodes] = 1.0
            return {"x": x, "adj": adj, "mask": mask}

        processed = {
            "train": [mol_to_data(mols[i]) for i in train_idx if mol_to_data(mols[i])],
            "val": [mol_to_data(mols[i]) for i in val_idx if mol_to_data(mols[i])],
            "test": [mol_to_data(mols[i]) for i in test_idx if mol_to_data(mols[i])]
        }
        
        import pickle
        with open(output_path, "wb") as f:
            pickle.dump(processed, f)

    def _process_csv_data(self, csv_path, output_path):
        df = pd.read_csv(csv_path)
        # We'll use the first 1000 for verify if we're just testing
        smiles_col = "SMILES1" if "SMILES1" in df.columns else "smiles"
        smiles_list = df[smiles_col].tolist()
        mols = [Chem.MolFromSmiles(s) for s in tqdm(smiles_list[:5000], desc="Processing CSV")]
        mols = [m for m in mols if m is not None]
        
        # Reuse _process_raw_logic by mocking suppl
        self._save_mols_to_pkl(mols, output_path)

    def _save_mols_to_pkl(self, mols, output_path):
        n = len(mols)
        indices = np.random.permutation(n)
        train_idx = indices[:int(0.8*n)]
        val_idx = indices[int(0.8*n):int(0.9*n)]
        test_idx = indices[int(0.9*n):]
        
        def mol_to_data(mol):
            n_nodes = mol.GetNumAtoms()
            if n_nodes > self.max_node_num: return None
            x = torch.zeros(self.max_node_num, dtype=torch.long)
            try:
                for i, atom in enumerate(mol.GetAtoms()):
                    x[i] = self.atom_encoder.get(atom.GetSymbol(), 0)
            except: return None
            
            adj = torch.zeros(self.max_node_num, self.max_node_num, dtype=torch.long)
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bt = bond.GetBondType()
                for k, bt_type in enumerate(self.bond_types):
                    if bt == bt_type:
                        adj[i, j] = adj[j, i] = k + 1
                        break
            mask = torch.zeros(self.max_node_num)
            mask[:n_nodes] = 1.0
            return {"x": x, "adj": adj, "mask": mask}

        processed = {
            "train": [mol_to_data(m) for m in [mols[i] for i in train_idx] if mol_to_data(m)],
            "val": [mol_to_data(m) for m in [mols[i] for i in val_idx] if mol_to_data(m)],
            "test": [mol_to_data(m) for m in [mols[i] for i in test_idx] if mol_to_data(m)]
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(processed, f)

    def _setup_pyg_data(self):
        pyg_ds = PyGQM9(root=os.path.join(self.cfg.data.dir, "pyg"))
        
        def pyg_to_data(data):
            # PyG QM9 has x [N, 11], edge_index [2, E], edge_attr [E, 4]
            n_nodes = data.x.shape[0]
            if n_nodes > self.max_node_num: return None
            
            # Map PyG atoms to our internal subset [C, N, O, F]
            # PyG order: H(0), C(1), N(2), O(3), F(4) ...
            # Our order: C(0), N(1), O(2), F(3)
            z = torch.argmax(data.x[:, :5], dim=-1) # Atomic numbers simplified
            x = torch.zeros(self.max_node_num, dtype=torch.long)
            for i in range(n_nodes):
                atom_idx = z[i].item() # 1=C, 2=N, 3=O, 4=F
                if atom_idx == 0: continue # Skip H if we remove H
                x[i] = max(0, atom_idx - 1)
                
            adj = torch.zeros(self.max_node_num, self.max_node_num, dtype=torch.long)
            edge_index = data.edge_index
            edge_attr = torch.argmax(data.edge_attr, dim=-1) + 1 # 1=S, 2=D, 3=T, 4=A
            for k in range(edge_index.shape[1]):
                i, j = edge_index[0, k], edge_index[1, k]
                adj[i, j] = edge_attr[k]
                
            mask = torch.zeros(self.max_node_num)
            mask[:n_nodes] = 1.0
            return {"x": x, "adj": adj, "mask": mask}

        # Convert a subset for testing
        subset_size = 5000
        indices = torch.randperm(len(pyg_ds))[:subset_size]
        data_list = [pyg_to_data(pyg_ds[i]) for i in indices if pyg_to_data(pyg_ds[i])]
        
        n = len(data_list)
        self.train_ds = self._build_dataset(data_list[:int(0.8*n)])
        self.val_ds = self._build_dataset(data_list[int(0.8*n):int(0.9*n)])
        self.test_ds = self._build_dataset(data_list[int(0.9*n):])

    def _build_dataset(self, data_list):
        X = torch.stack([d["x"] for d in data_list])
        E = torch.stack([d["adj"] for d in data_list])
        M = torch.stack([d["mask"] for d in data_list])
        
        # Convert to one-hot for node features
        X_one_hot = torch.nn.functional.one_hot(X, num_classes=self.num_atom_types).float()
        # Edge features remain as indices or can be one-hot
        # In this repo, trainers often expect (features, adjs, masks)
        return TensorDataset(X_one_hot, E.float(), M)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

    def node_counts(self, max_nodes_possible=None):
        if max_nodes_possible is None:
            max_nodes_possible = self.max_node_num + 1
        all_counts = torch.zeros(max_nodes_possible)
        for batch in self.train_dataloader():
            masks = batch[2]
            counts = masks.sum(dim=1).long()
            for c in counts:
                if c < max_nodes_possible:
                    all_counts[c] += 1
        return all_counts / all_counts.sum()

    def train_smiles(self):
        """Return a set of canonical SMILES strings for the training set (used for novelty)."""
        if self._train_smiles_cache is not None:
            return self._train_smiles_cache

        from src.metrics.mol_metrics import build_molecule, mol2smiles
        smiles_set = set()
        for batch in tqdm(self.train_dataloader(), desc="Precomputing train SMILES"):
            X_batch, E_batch, M_batch = batch
            # X_batch is one-hot [B, N, num_atom_types+1], take argmax to get indices
            X_idx = X_batch.argmax(dim=-1)  # [B, N]
            E_idx = E_batch.long()           # [B, N, N]
            for i in range(X_idx.shape[0]):
                n = int(M_batch[i].sum().item())
                x = X_idx[i, :n]
                e = E_idx[i, :n, :n]
                mol = build_molecule(x, e, self.atom_decoder)
                smi = mol2smiles(mol)
                if smi is not None:
                    smiles_set.add(smi)
        self._train_smiles_cache = smiles_set
        return smiles_set
