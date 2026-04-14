import torch
from src.utils import PlaceHolder

class ExtraMolecularFeatures:
    def __init__(self, dataset_info):
        self.charge = ChargeFeature(dataset_info=dataset_info)
        self.valency = ValencyFeature()
        self.weight = WeightFeature(dataset_info=dataset_info)

    def __call__(self, noisy_data):
        charge = self.charge(noisy_data).unsqueeze(-1)  # (bs, n, 1)
        valency = self.valency(noisy_data).unsqueeze(-1)  # (bs, n, 1)
        weight = self.weight(noisy_data)  # (bs, 1)

        E = noisy_data["E_t"]
        extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)

        return PlaceHolder(
            X=torch.cat((charge, valency), dim=-1), 
            E=extra_edge_attr, 
            y=weight
        )

class ChargeFeature:
    def __init__(self, dataset_info):
        self.valencies = dataset_info.valencies if hasattr(dataset_info, "valencies") else [4, 3, 2, 1]

    def __call__(self, noisy_data):
        E_t = noisy_data["E_t"]
        de = E_t.shape[-1]
        if de == 5:
            bond_orders = torch.tensor([0, 1, 2, 3, 1.5], device=E_t.device).reshape(1, 1, 1, -1)
        else:
            bond_orders = torch.tensor([0, 1, 2, 3], device=E_t.device).reshape(1, 1, 1, -1)
        
        weighted_E = E_t * bond_orders  # (bs, n, n, de)
        current_valencies = weighted_E.sum(dim=-1).sum(dim=-1)  # (bs, n)
        
        # We need the one-hot noisy atoms
        X_t = noisy_data["X_t"]
        dx = X_t.shape[-1]
        
        # Pad or trim valencies to match exactly dx atom types
        val = list(self.valencies)
        while len(val) < dx:
            val.append(0)
        val = val[:dx]
            
        valencies = torch.tensor(val, device=X_t.device).reshape(1, 1, -1)
        X = X_t * valencies  # (bs, n, dx)
        normal_valencies = torch.sum(X, dim=-1)  # (bs, n) assuming X is one-hot probabilities
        
        return (normal_valencies - current_valencies).type_as(X_t)


class ValencyFeature:
    def __call__(self, noisy_data):
        E_t = noisy_data["E_t"]
        de = E_t.shape[-1]
        if de == 5:
            bond_orders = torch.tensor([0, 1, 2, 3, 1.5], device=E_t.device).reshape(1, 1, 1, -1)
        else:
            bond_orders = torch.tensor([0, 1, 2, 3], device=E_t.device).reshape(1, 1, 1, -1)
        
        E = E_t * bond_orders  # (bs, n, n, de)
        valencies = E.sum(dim=-1).sum(dim=-1)  # (bs, n)
        return valencies.type_as(noisy_data["X_t"])

class WeightFeature:
    def __init__(self, dataset_info):
        self.max_weight = getattr(dataset_info, "max_weight", 390)
        self.atom_weights = getattr(dataset_info, "atom_weights", {0: 1, 1: 12, 2: 14, 3: 16, 4: 19})

    def __call__(self, noisy_data):
        X_t = noisy_data["X_t"]  # (bs, n, dx)
        
        weights = [self.atom_weights.get(i, 0) for i in range(X_t.shape[-1])]
        weights_tensor = torch.tensor(weights, device=X_t.device).reshape(1, 1, -1)
        
        X_weights = (X_t * weights_tensor).sum(dim=-1)  # (bs, n)
        return (X_weights.sum(dim=-1).unsqueeze(-1).type_as(X_t) / self.max_weight)
