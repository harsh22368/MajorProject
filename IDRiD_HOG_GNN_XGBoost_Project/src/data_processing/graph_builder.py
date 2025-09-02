"""Graph Builder for GNN processing"""
import torch
import numpy as np
from torch_geometric.data import Data
from torch_cluster import knn_graph
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class IDRiDGraphBuilder:
    """Build spatial graphs from HOG features"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.graph_config = config['gnn']['graph_construction']
        self.k_neighbors = self.graph_config['k_neighbors']
        self.distance_metric = self.graph_config['distance_metric']
        
        logger.info(f"Graph Builder initialized with k={self.k_neighbors}")
    
    def build_spatial_graph(self, hog_features: np.ndarray, 
                           coordinates: np.ndarray,
                           image_id: str = "",
                           labels: Optional[np.ndarray] = None) -> Data:
        """Build spatial graph from HOG features"""
        if len(hog_features) == 0:
            return self._create_dummy_graph()
        
        x = torch.tensor(hog_features, dtype=torch.float32)
        pos = torch.tensor(coordinates, dtype=torch.float32)
        
        try:
            if len(coordinates) <= self.k_neighbors:
                edge_index = self._create_fully_connected_graph(len(coordinates))
            else:
                edge_index = knn_graph(pos, k=self.k_neighbors, loop=False)
        except Exception as e:
            logger.warning(f"Error creating k-NN graph: {e}")
            edge_index = self._create_fully_connected_graph(len(coordinates))
        
        data = Data(x=x, pos=pos, edge_index=edge_index)
        data.image_id = image_id
        data.num_nodes = x.size(0)
        data.num_edges = edge_index.size(1)
        
        if labels is not None:
            data.y = torch.tensor(labels, dtype=torch.long)
        
        return self._add_edge_attributes(data)
    
    def _create_dummy_graph(self) -> Data:
        """Create dummy graph for edge cases"""
        x = torch.zeros((1, 36), dtype=torch.float32)
        pos = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        data = Data(x=x, pos=pos, edge_index=edge_index)
        data.num_nodes = 1
        data.num_edges = 1
        data.image_id = "dummy"
        return data
    
    def _create_fully_connected_graph(self, num_nodes: int) -> torch.Tensor:
        """Create fully connected graph"""
        if num_nodes <= 1:
            return torch.tensor([[0], [0]], dtype=torch.long)
        
        edges = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
        if not edges:
            return torch.tensor([[0], [0]], dtype=torch.long)
        
        return torch.tensor(edges, dtype=torch.long).t()
    
    def _add_edge_attributes(self, data: Data) -> Data:
        """Add edge attributes like distance"""
        pos = data.pos
        edge_index = data.edge_index
        
        if edge_index.size(1) == 0:
            data.edge_attr = torch.empty((0, 2), dtype=torch.float32)
            return data
        
        try:
            row, col = edge_index
            edge_vectors = pos[col] - pos[row]
            edge_distances = torch.norm(edge_vectors, dim=1)
            edge_angles = torch.atan2(edge_vectors[:, 1], edge_vectors[:, 0])
            
            data.edge_attr = torch.stack([edge_distances, edge_angles], dim=1)
        except Exception:
            data.edge_attr = torch.zeros((edge_index.size(1), 2), dtype=torch.float32)
        
        return data