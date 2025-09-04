"""Graph Neural Network models"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class HOGGraphNet(nn.Module):
    """GNN for processing HOG feature graphs"""

    def __init__(self, config: Dict):
        super(HOGGraphNet, self).__init__()

        self.config = config
        self.gnn_config = config['gnn']

        self.input_dim = self.gnn_config['input_dim']
        self.hidden_dims = self.gnn_config['hidden_dims']
        self.num_layers = self.gnn_config['num_layers']
        self.dropout = self.gnn_config['dropout']
        self.activation = self.gnn_config.get('activation', 'relu')

        # Build GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        current_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            self.convs.append(GCNConv(current_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim

        # Attention layer
        self.attention = GATConv(
            current_dim, current_dim, heads=4, concat=False, dropout=self.dropout
        )

        self.final_dim = current_dim
        self.dropout_layer = nn.Dropout(self.dropout)
        self.activation_fn = F.relu if self.activation == 'relu' else F.elu

        logger.info(f"GNN initialized: {self.input_dim} -> {self.hidden_dims} -> {self.final_dim}")

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through GNN"""
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        # Apply GCN layers
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.activation_fn(x)
            x = self.dropout_layer(x)

        # Apply attention
        x = self.attention(x, edge_index)
        x = self.activation_fn(x)

        node_embeddings = x

        # Global pooling
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        graph_embedding = torch.cat([mean_pool, max_pool], dim=1)

        return graph_embedding, node_embeddings

    def get_embedding_dim(self) -> int:
        """Get graph embedding dimension"""
        return self.final_dim * 2


class GNNWithFeatureExtraction(nn.Module):
    """GNN for feature extraction for XGBoost"""

    def __init__(self, config: Dict):
        super(GNNWithFeatureExtraction, self).__init__()

        self.gnn_encoder = HOGGraphNet(config)
        embedding_dim = self.gnn_encoder.get_embedding_dim()

        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )

        logger.info("Feature extraction GNN initialized")

    def forward(self, data: Data) -> torch.Tensor:
        """Extract features for XGBoost"""
        graph_embedding, _ = self.gnn_encoder(data)
        features = self.feature_extractor(graph_embedding)
        return features

    def extract_features(self, data: Data) -> torch.Tensor:
        """Extract features/embeddings from graph data - ADDED METHOD"""
        self.eval()
        with torch.no_grad():
            # Use the existing forward method
            embeddings = self.forward(data)

            # If forward returns a tuple, get the embeddings part
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]

            # Return as tensor for compatibility
            return embeddings


def create_gnn_model(config: Dict, model_type: str = "feature_extraction") -> nn.Module:
    """Factory function for GNN models"""
    if model_type == "base":
        return HOGGraphNet(config)
    elif model_type == "feature_extraction":
        return GNNWithFeatureExtraction(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
