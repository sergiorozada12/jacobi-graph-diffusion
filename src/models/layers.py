import torch
import torch.nn.functional as F
from torch.nn import Parameter
import math

from src.utils import mask_adjs, mask_x


class DenseGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        self._glorot(self.weight)
        self._zeros(self.bias)

    def _glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def _zeros(self, tensor):
        if tensor is not None:
            tensor.data.zero_()

    def forward(self, x, adj, mask=None, add_loop=True):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, device=adj.device)
            adj[:, idx, idx] = 2 if self.improved else 1

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out += self.bias

        if mask is not None:
            out *= mask.view(B, N, 1).to(out.dtype)

        return out


class MLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, use_bn=False, activate_func=F.relu):
        super().__init__()
        self.linear_or_not = num_layers == 1
        self.use_bn = use_bn
        self.activate_func = activate_func

        if self.linear_or_not:
            self.linear = torch.nn.Linear(input_dim, output_dim)
        else:
            self.linears = torch.nn.ModuleList()
            self.linears.append(torch.nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(torch.nn.Linear(hidden_dim, output_dim))

            if use_bn:
                self.batch_norms = torch.nn.ModuleList(
                    [torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)]
                )

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)

        h = x
        for i, layer in enumerate(self.linears[:-1]):
            h = layer(h)
            if self.use_bn:
                h = self.batch_norms[i](h)
            h = self.activate_func(h)
        return self.linears[-1](h)


class Attention(torch.nn.Module):
    def __init__(self, in_dim, attn_dim, out_dim, num_heads=4, conv='GCN'):
        super().__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.out_dim = out_dim
        self.conv = conv
        self.activation = torch.tanh
        self.softmax_dim = 2

        self.gnn_q, self.gnn_k, self.gnn_v = self._get_gnn(in_dim, attn_dim, out_dim, conv)

    def forward(self, x, adj, flags, attention_mask=None):
        Q = self.gnn_q(x, adj) if self.conv == 'GCN' else self.gnn_q(x)
        K = self.gnn_k(x, adj) if self.conv == 'GCN' else self.gnn_k(x)
        V = self.gnn_v(x, adj)

        dim_split = self.attn_dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, dim=-1), dim=0)
        K_ = torch.cat(K.split(dim_split, dim=-1), dim=0)

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask] * self.num_heads, dim=0)
            scores = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.out_dim)
            A = self.activation(attention_mask + scores)
        else:
            A = self.activation(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.out_dim))

        A = A.view(-1, *adj.shape)
        A = (A.mean(dim=0) + A.mean(dim=0).transpose(-1, -2)) / 2

        return V, A

    def _get_gnn(self, in_dim, attn_dim, out_dim, conv='GCN'):
        if conv == 'GCN':
            return (
                DenseGCNConv(in_dim, attn_dim),
                DenseGCNConv(in_dim, attn_dim),
                DenseGCNConv(in_dim, out_dim)
            )
        elif conv == 'MLP':
            num_layers = 2
            return (
                MLP(num_layers, in_dim, 2 * attn_dim, attn_dim, activate_func=torch.tanh),
                MLP(num_layers, in_dim, 2 * attn_dim, attn_dim, activate_func=torch.tanh),
                DenseGCNConv(in_dim, out_dim)
            )
        else:
            raise NotImplementedError(f'{conv} not implemented.')


class AttentionLayer(torch.nn.Module):
    def __init__(self, num_linears, conv_input_dim, attn_dim, conv_output_dim, input_dim, output_dim, num_heads=4, conv='GCN'):
        super().__init__()
        self.attn = torch.nn.ModuleList([
            Attention(conv_input_dim, attn_dim, conv_output_dim, num_heads=num_heads, conv=conv)
            for _ in range(input_dim)
        ])
        hidden_dim = 2 * max(input_dim, output_dim)

        self.mlp = MLP(num_linears, 2 * input_dim, hidden_dim, output_dim, use_bn=False, activate_func=F.elu)
        self.multi_channel = MLP(2, input_dim * conv_output_dim, hidden_dim, conv_output_dim, use_bn=False, activate_func=F.elu)

    def forward(self, x, adj, flags):
        x_list = []
        mask_list = []
        for i, attn in enumerate(self.attn):
            _x, mask = attn(x, adj[:, i, :, :], flags)
            x_list.append(_x)
            mask_list.append(mask.unsqueeze(-1))

        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)), flags)
        x_out = torch.tanh(x_out)

        mlp_in = torch.cat([torch.cat(mask_list, dim=-1), adj.permute(0, 2, 3, 1)], dim=-1)
        B, N, _, _ = mlp_in.shape
        mlp_out = self.mlp(mlp_in.view(-1, mlp_in.shape[-1]))
        adj_out = mlp_out.view(B, N, N, -1).permute(0, 3, 1, 2)
        adj_out = adj_out + adj_out.transpose(-1, -2)
        adj_out = mask_adjs(adj_out, flags)

        return x_out, adj_out
