import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import APPNP as APPNPConv
from torch_geometric.nn import SAGEConv

from .custom_layers import MultiplicationLayer


class GraphSAGE(nn.Module):
    def __init__(self, num_features, num_classes, embedding_dim) -> None:
        super().__init__()
        self.conv = SAGEConv(num_features, embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        embedding = self.conv(x, edge_index)
        logits = F.relu(embedding)
        logits = F.dropout(logits, training=self.training)
        logits = self.linear(logits)
        return embedding, logits


class APPNP(nn.Module):
    def __init__(self, num_features, num_classes, embedding_dim) -> None:
        super().__init__()
        self.conv = APPNPConv(K=3, alpha=0.5)
        self.linear1 = nn.Linear(num_features, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        embedding = self.conv(x, edge_index)
        embedding = F.relu(embedding)
        embedding = F.dropout(embedding, training=self.training)
        embedding = self.linear1(embedding)
        logits = F.relu(embedding)
        logits = F.dropout(logits, training=self.training)
        logits = self.linear2(logits)
        return embedding, logits


class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        x = F.dropout(x, training=self.training)
        return x


class PostHocLearnedWeightsClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, num_levels):
        super().__init__()
        self.post_hoc_weights = nn.ModuleList(
            [MultiplicationLayer(embedding_dim) for _ in range(num_levels)])
        self.linear = nn.Linear(embedding_dim, num_classes)
        self.num_levels = num_levels

    def forward(self, x):
        transformed_embeddings = []
        for level in range(self.num_levels):
            transformed_embeddings.append(
                self.post_hoc_weights[level](x[:, level])
            )
        post_hoc_learned_embedding = torch.stack(
            transformed_embeddings).sum(dim=0)
        out = self.linear(post_hoc_learned_embedding)
        out = F.dropout(out, training=self.training)
        return out
