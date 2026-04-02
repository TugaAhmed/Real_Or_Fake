import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_max_pool as gmp, global_mean_pool as gap

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=1, heads=8, dropout=0.3):
        super().__init__()

        # GAT layers with multi-head attention
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)

        # Batch normalization after each GAT layer
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Linear layers
        # lin0 takes mean+max concatenated → hidden_channels*2
        self.lin_news = nn.Linear(in_channels, hidden_channels)
        self.lin0 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin1 = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Message passing with BN + dropout
        h = self.dropout(self.bn1(self.conv1(x, edge_index).relu()))
        h = self.dropout(self.bn2(self.conv2(h, edge_index).relu()))
        h = self.dropout(self.bn3(self.conv3(h, edge_index).relu()))

        # Mean + Max pooling combined
        h = torch.cat([gmp(h, batch), gap(h, batch)], dim=-1)  # hidden*2
        h = self.lin0(h).relu()

        # Root node extraction (unchanged)
        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat([root.new_zeros(1), root + 1], dim=0)
        news = x[root]
        news = self.lin_news(news).relu()

        out = self.lin1(torch.cat([h, news], dim=-1))
        return torch.sigmoid(out)