from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.attention import SGFormerAttention
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import to_dense_batch

class GraphBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        dropout
    ):
        super().__init__()

        self.conv = GCNConv(hidden_channels, hidden_channels)
        self.bn = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout = dropout
        self.activation = F.relu

    def forward(self, x, last_x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + last_x
        return x


class GraphModule(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        dropout=0.5,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.fcs = torch.nn.ModuleList()
        self.fcs.append(torch.nn.Linear(in_channels, hidden_channels))
        self.initial_bn = torch.nn.BatchNorm1d(hidden_channels)

        self.blocks = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(GraphBlock(hidden_channels, dropout))

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        self.initial_bn.reset_parameters()
        for block in self.blocks:
            block.conv.reset_parameters()
            block.bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        x = self.fcs[0](x)
        x = self.initial_bn(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        last_x = x
        for block in self.blocks:
            x = block(x, last_x, edge_index)
            last_x = x

        return x

    
class SGBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, dropout):
        super().__init__()
        self.dropout = dropout
        self.attn = SGFormerAttention(hidden_channels, num_heads, hidden_channels)
        self.bn = torch.nn.LayerNorm(hidden_channels)
        self.activation = F.relu

    def forward(self, x, last_x, mask):
        x = self.attn(x, mask)
        x = (x + last_x) / 2.
        x = self.bn(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x
    

class SGModule(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        num_heads=1,
        dropout=0.5,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.fcs = torch.nn.ModuleList()
        self.fcs.append(torch.nn.Linear(in_channels, hidden_channels))
        self.initial_bn = torch.nn.LayerNorm(hidden_channels)
        self.blocks = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(SGBlock(hidden_channels, num_heads, dropout))

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        self.initial_bn.reset_parameters()
        for block in self.blocks:
            block.attn.reset_parameters()
            block.bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x: Tensor, batch: Tensor):
        # to dense batch expects sorted batch
        batch, indices = batch.sort(stable=True)
        rev_perm = torch.empty_like(indices)
        rev_perm[indices] = torch.arange(len(indices), device=indices.device)
        x = x[indices]
        x, mask = to_dense_batch(x, batch)

        # input MLP layer
        x = self.fcs[0](x)
        x = self.initial_bn(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link  
        last_x = x
        for block in self.blocks:
            x = block(x, last_x, mask)
            last_x = x

        x_mask = x[mask]
        # reverse the sorting
        unsorted_x_mask = x_mask[rev_perm]
        return unsorted_x_mask


class SGFormer(torch.nn.Module):
    r"""The sgformer module from the
    `"SGFormer: Simplifying and Empowering Transformers for
    Large-Graph Representations"
    <https://arxiv.org/abs/2306.10759>`_ paper.

    Args:
        in_channels (int): Input channels.
        hidden_channels (int): Hidden channels.
        out_channels (int): Output channels.
        trans_num_layers (int): The number of layers for all-pair attention.
            (default: :obj:`2`)
        trans_num_heads (int): The number of heads for attention.
            (default: :obj:`1`)
        trans_dropout (float): Global dropout rate.
            (default: :obj:`0.5`)
        gnn_num_layers (int): The number of layers for GNN.
            (default: :obj:`3`)
        gnn_dropout (float): GNN dropout rate.
            (default: :obj:`0.5`)
        graph_weight (float): The weight balance global and gnn module.
            (default: :obj:`0.5`)
        aggregate (str): Aggregate type.
            (default: :obj:`add`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        trans_num_layers: int = 2,
        trans_num_heads: int = 1,
        trans_dropout: float = 0.5,
        gnn_num_layers: int = 3,
        gnn_dropout: float = 0.5,
        graph_weight: float = 0.5,
        aggregate: str = 'add',
    ):
        super().__init__()
        self.trans_conv = SGModule(
            in_channels,
            hidden_channels,
            trans_num_layers,
            trans_num_heads,
            trans_dropout,
        )
        self.graph_conv = GraphModule(
            in_channels,
            hidden_channels,
            gnn_num_layers,
            gnn_dropout,
        )
        self.graph_weight = graph_weight

        self.aggregate = aggregate

        if aggregate == 'add':
            self.fc = torch.nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = torch.nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.graph_conv.parameters())
        self.params2.extend(list(self.fc.parameters()))

        self.out_channels = out_channels

    def reset_parameters(self) -> None:
        self.trans_conv.reset_parameters()
        self.graph_conv.reset_parameters()
        self.fc.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
        """
        x1 = self.trans_conv(x, batch)
        x2 = self.graph_conv(x, edge_index)
        if self.aggregate == 'add':
            x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
        else:
            x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
