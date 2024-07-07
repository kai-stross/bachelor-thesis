from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import numpy as np

class GCN_2_Layers(nn.Module):
    def __init__(self, num_features, out_channels, dropout_rate):
        super(GCN_2_Layers, self).__init__()
        # First GCN layer with hidden layer channels
        self.conv1 = GCNConv(num_features, out_channels)
        # Second GCN layer with 2 output layer channels for binary classification
        self.conv2 = GCNConv(out_channels, 2)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        # First convolutional layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # Second convolutional layer
        x = self.conv2(x, edge_index)
        return x
class GCN_3_Layers(nn.Module):
    def __init__(self, num_features, out_channels, dropout_rate):
        super(GCN_3_Layers, self).__init__()
        # First GCN layer
        self.conv1 = GCNConv(num_features, out_channels)
        # Second GCN layer
        self.conv2 = GCNConv(out_channels, out_channels)
        # Third GCN layer also with 2 output channels for binary classification
        self.conv3 = GCNConv(out_channels, 2)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        # First convolutional layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # Second convolutional layer with ReLU activation
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # Third convolutional layer
        x = self.conv3(x, edge_index)
        return x

class GeneralLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, activation_fn):
        super(GeneralLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation_fn

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x

class GNNNodeHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNNodeHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class GNNStackStage(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, activation_fn, layers):
        super(GNNStackStage, self).__init__()
        self.layers = nn.ModuleList()

        if isinstance(in_channels, np.integer):
            in_channels = int(in_channels)

        for _ in range(layers):
            sage_layer = SAGEConv(in_channels=in_channels, out_channels=out_channels, aggr='mean')
            post_layer = GeneralLayer(out_channels, out_channels, dropout, activation_fn)
            self.layers.append(nn.Sequential(sage_layer, post_layer))
            in_channels = out_channels
    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer[0](x, edge_index)
            x = layer[1](x)
        return x

class GraphSAGE(nn.Module):
    def __init__(self, num_features, out_channels, dropout_rate):
        super().__init__()
        activation_fn = nn.PReLU()
        self.pre_mp1 = GeneralLayer(num_features, out_channels, dropout_rate, activation_fn)
        self.pre_mp2 = GeneralLayer(out_channels, out_channels, dropout_rate, activation_fn)
        self.mp = GNNStackStage(in_channels=out_channels, out_channels=out_channels, dropout=dropout_rate, activation_fn=activation_fn, layers=3)
        self.post_mp = GNNNodeHead(out_channels, 2)

    def forward(self, x, edge_index):
        x = self.pre_mp1(x)
        x = self.pre_mp2(x)
        x = self.mp(x, edge_index)
        x = self.post_mp(x)
        return x