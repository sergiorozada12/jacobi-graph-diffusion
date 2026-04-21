import re
import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# Atom valency and bond type mappings consistent with DeFoG/DiGress
bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# Atomic numbers for QM9 atoms
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

class BasicMolecularMetrics:
    def __init__(self, dataset_info, train_smiles=None):
        self.atom_decoder = dataset_info.atom_decoder
        self.train_smiles = train_smiles

    def reset(self):
        pass

    def forward(self, generated, ref_metrics=None, local_rank=0, test=False):
        """
        generated: list of PlaceHolder(X, E)
        """
        # Convert PlaceHolder list to (atom_types, edge_types) tuples
        mol_list = []
        for p in generated:
            # p.X is [B, N], p.E is [B, N, N]
            B = p.X.shape[0]
            for i in range(B):
                mol_list.append((p.X[i], p.E[i]))
        
        results, _ = self.evaluate(mol_list)
        return results

    def compute_validity(self, generated, relaxed=False):
        """
        generated: list of tuples (atom_types, edge_types)
        """
        valid = []
        all_smiles = []
        num_components = []
        lcc_fractions = []
        
        for graph in tqdm(generated, desc="Computing validity" if not relaxed else "Computing relaxed validity"):
            atom_types, edge_types = graph
            if relaxed:
                mol = build_molecule_with_partial_charges(atom_types, edge_types, self.atom_decoder)
            else:
                mol = build_molecule(atom_types, edge_types, self.atom_decoder)
            
            # Basic sanitization and SMILES
            smiles = mol2smiles(mol)
            
            if smiles is not None:
                # Handle fragments
                try:
                    frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                    num_components.append(len(frags))
                    if len(frags) > 1:
                        largest_mol = max(frags, key=lambda m: m.GetNumAtoms())
                        smiles = mol2smiles(largest_mol)
                        lcc_fractions.append(largest_mol.GetNumAtoms() / mol.GetNumAtoms())
                    else:
                        lcc_fractions.append(1.0)
                except:
                    num_components.append(1)
                    lcc_fractions.append(1.0)
                
                valid.append(smiles)
                all_smiles.append(smiles)
            else:
                all_smiles.append(None)
                num_components.append(0)
                lcc_fractions.append(0.0)
        
        validity_score = len(valid) / len(generated) if len(generated) > 0 else 0
        return valid, validity_score, all_smiles, {
            "num_components": np.mean(num_components) if num_components else 0,
            "lcc_fraction": np.mean(lcc_fractions) if lcc_fractions else 0
        }

    def compute_uniqueness(self, valid_smiles):
        if len(valid_smiles) == 0:
            return [], 0.0
        unique_smiles = list(set(valid_smiles))
        return unique_smiles, len(unique_smiles) / len(valid_smiles)

    def compute_novelty(self, unique_smiles):
        if self.train_smiles is None or len(unique_smiles) == 0:
            return [], 1.0
        novel_smiles = [s for s in unique_smiles if s not in self.train_smiles]
        return novel_smiles, len(novel_smiles) / len(unique_smiles)

    def evaluate(self, generated):
        # Strict metrics
        strict_valid_smiles, strict_validity, _, strict_info = self.compute_validity(generated, relaxed=False)
        strict_unique_smiles, strict_uniqueness = self.compute_uniqueness(strict_valid_smiles)
        _, strict_novelty = self.compute_novelty(strict_unique_smiles)
        
        # Relaxed metrics
        relaxed_valid_smiles, relaxed_validity, _, relaxed_info = self.compute_validity(generated, relaxed=True)
        relaxed_unique_smiles, relaxed_uniqueness = self.compute_uniqueness(relaxed_valid_smiles)
        _, relaxed_novelty = self.compute_novelty(relaxed_unique_smiles)
        
        return {
            "validity": strict_validity,
            "uniqueness": strict_uniqueness,
            "novelty": strict_novelty,
            "relaxed_validity": relaxed_validity,
            "relaxed_uniqueness": relaxed_uniqueness,
            "relaxed_novelty": relaxed_novelty,
            "mean_components": relaxed_info["num_components"],
            "mean_lcc_fraction": relaxed_info["lcc_fraction"]
        }, relaxed_valid_smiles

def mol2smiles(mol):
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)

def build_molecule(atom_types, edge_types, atom_decoder):
    """
    atom_types: [N] (indices)
    edge_types: [N, N] (indices 0..4)
    """
    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
    
    edge_types = torch.triu(edge_types, diagonal=1)
    all_bonds = torch.nonzero(edge_types)
    for bond in all_bonds:
        u, v = bond[0].item(), bond[1].item()
        bond_type_idx = edge_types[u, v].item()
        if bond_type_idx > 0 and bond_type_idx < len(bond_dict):
            mol.AddBond(u, v, bond_dict[int(bond_type_idx)])
            
    return mol

def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence

def build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder):
    """
    Ported from DeFoG to handle formal charges correctly.
    """
    mol = build_molecule(atom_types, edge_types, atom_decoder)
    flag, atomid_valence = check_valency(mol)
    if not flag:
        # Simple heuristic for formal charges on N, O, S
        if len(atomid_valence) == 2:
            idx, v = atomid_valence[0], atomid_valence[1]
            atom = mol.GetAtomWithIdx(idx)
            atomic_num = atom.GetAtomicNum()
            if atomic_num in (7, 8, 16) and (v - ATOM_VALENCY.get(atomic_num, 0)) == 1:
                atom.SetFormalCharge(1)
    return mol
