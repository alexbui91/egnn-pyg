import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

class EGNNConv(MessagePassing):
    r"""
    Equivariant Graph Convolutional Layer from `E(n) Equivariant Graph Neural Networks <https://arxiv.org/abs/2102.09844>`.
    
    Parameters
    ----------
    in_size : int
        Input feature size (node features).
    hidden_size : int
        Hidden feature size for MLPs.
    out_size : int
        Output feature size (node features).
    edge_feat_size : int, optional
        Edge feature size; default is 0.
    """
    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0):
        super(EGNNConv, self).__init__(aggr='mean')  # Use mean aggregation

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # φ_e
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )

        # φ_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size),
        )

        # φ_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, x, edge_index, edge_attr=None, pos=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (N, in_size).
        edge_index : torch.Tensor
            Edge indices of shape (2, E).
        edge_attr : torch.Tensor, optional
            Edge features of shape (E, edge_feat_size); default is None.
        pos : torch.Tensor
            Node positions of shape (N, 3).

        Returns
        -------
        x_out : torch.Tensor
            Updated node features of shape (N, out_size).
        pos_out : torch.Tensor
            Updated node positions of shape (N, 3).
        """
        x_out, pos_out = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr, size=None)

        return x_out, pos_out
    
    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        # Compute coordinate differences and radial distances
        coord_diff = pos_i - pos_j
        radial = (coord_diff ** 2).sum(dim=-1, keepdim=True)

        # Concatenate features for edge MLP
        if edge_attr is not None:
            edge_features = torch.cat([x_i, x_j, radial, edge_attr], dim=-1)
        else:
            edge_features = torch.cat([x_i, x_j, radial], dim=-1)

        # Compute messages
        msg_h = self.edge_mlp(edge_features)
        msg_x = self.coord_mlp(msg_h) * coord_diff / (radial.sqrt() + 1e-30)
        return msg_h, msg_x
    
    def aggregate(self, inputs, index, dim_size=None):
        # Separate node and position messages
        msg_h, msg_x = inputs

        # Aggregate node features
        h_neigh = scatter(msg_h, index, dim=self.node_dim, reduce='mean', dim_size=dim_size)

        # Aggregate position updates
        pos_neigh = scatter(msg_x, index, dim=self.node_dim, reduce='mean', dim_size=dim_size)

        return h_neigh, pos_neigh
    
    def update(self, aggr_out, x, pos):
        # Unpack aggregated messages
        h_neigh, pos_neigh = aggr_out

        # update node features
        h_out = self.node_mlp(torch.cat([x, h_neigh], dim=-1))

        # Update node positions
        pos_out = pos + pos_neigh

        return h_out, pos_out