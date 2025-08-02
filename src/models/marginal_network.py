import torch
import torch.nn.functional as F

from src.models.layers import MLP
from src.models.layers import AttentionLayer
from src.utils import mask_adjs, mask_x


class MarginalNetwork(torch.nn.Module):
    def __init__(self, max_feat_num, nhid, num_layers, num_linears,
                 c_init, c_hid, c_final, adim, num_heads=4, conv="GCN"):
        super().__init__()

        self.nfeat = max_feat_num
        self.nhid = nhid
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final
        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            c_in = c_init if i == 0 else c_hid
            c_out = c_final if i == num_layers - 1 else c_hid
            conv_in = max_feat_num if i == 0 else nhid
            conv_out = nhid

            self.layers.append(
                AttentionLayer(num_linears, conv_in, adim, conv_out, c_in, c_out, num_heads, conv)
            )

        self.fdim = c_hid * (num_layers - 1) + c_final + c_init

        self.final_adj = MLP(
            num_layers=3,
            input_dim=self.fdim,
            hidden_dim=2 * self.fdim,
            output_dim=1,
            use_bn=False,
            activate_func=F.elu
        )

        self.final_x = MLP(
            num_layers=3,
            input_dim=self.nhid,
            hidden_dim=2 * self.nhid,
            output_dim=self.nfeat,
            use_bn=False,
            activate_func=F.elu
        )

    def forward(self, x, adj, flags=None):
        B, N, F = x.size()
        B, N, _ = adj.size()

        adjc = self._pow_tensor(adj, self.c_init)
        adj_list = [adjc]

        for layer in self.layers:
            x, adjc = layer(x, adjc, flags)
            adj_list.append(adjc)

        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)  # [B, N, N, F]
        logits = self.final_adj(adjs).view(B, N, N)
        adj_pred = torch.sigmoid(logits)

        x = self.final_x(x).view(B, N, F)

        mask = torch.ones(N, N, device=adj.device) - torch.eye(N, device=adj.device)
        adj_pred = adj_pred * mask.unsqueeze(0)

        return mask_x(x, flags), mask_adjs(adj_pred, flags)

    def _pow_tensor(self, x, cnum):
        x_ = x.clone()
        xc = [x.unsqueeze(1)]
        for _ in range(cnum - 1):
            x_ = torch.bmm(x_, x)
            xc.append(x_.unsqueeze(1))
        return torch.cat(xc, dim=1)
