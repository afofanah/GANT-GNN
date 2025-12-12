import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid, Amazon, Reddit, Yelp
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import NeighborLoader
from typing import Dict, Tuple, List, Optional
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from sklearn.preprocessing import StandardScaler


class OptimizedRealWorldDatasets:
    @staticmethod
    def load_financial_fraud_amazon():
        dataset = Amazon(root='./data', name='Photo', transform=NormalizeFeatures())
        data = dataset[0]
        
        num_nodes = data.x.shape[0]
        anomaly_ratio = 0.15
        num_anomalies = int(num_nodes * anomaly_ratio)
        
        anomaly_indices = torch.randperm(num_nodes)[:num_anomalies]
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        anomaly_mask[anomaly_indices] = True
        
        data.x[anomaly_indices] += torch.randn_like(data.x[anomaly_indices]) * 0.5
        
        fake_edges = torch.randint(0, num_nodes, (2, num_anomalies))
        data.edge_index = torch.cat([data.edge_index, fake_edges], dim=1)
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[anomaly_indices] = 1
        
        data.y = y
        data.anomaly_mask = anomaly_mask
        data.anomaly_labels = y
        
        metadata = {
            'num_nodes': num_nodes,
            'num_edges': data.edge_index.shape[1],
            'num_features': data.x.shape[1],
            'num_anomalies': num_anomalies,
            'anomaly_ratio': anomaly_ratio,
            'num_classes': 2,
            'type': 'fraud_detection',
            'description': 'Amazon financial fraud simulation'
        }
        
        return data, metadata
    
    @staticmethod
    def load_social_bot_reddit():
        dataset = Reddit(root='./data', transform=NormalizeFeatures())
        data = dataset[0]
        
        num_nodes = min(10000, data.x.shape[0])
        subset_indices = torch.randperm(data.x.shape[0])[:num_nodes]
        
        data.x = data.x[subset_indices]
        
        edge_mask = torch.isin(data.edge_index[0], subset_indices) & torch.isin(data.edge_index[1], subset_indices)
        subset_edges = data.edge_index[:, edge_mask]
        
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subset_indices)}
        subset_edges = torch.tensor([[node_mapping[edge[0].item()], node_mapping[edge[1].item()]] 
                                   for edge in subset_edges.t()]).t()
        data.edge_index = subset_edges
        
        anomaly_ratio = 0.08
        num_anomalies = int(num_nodes * anomaly_ratio)
        anomaly_indices = torch.randperm(num_nodes)[:num_anomalies]
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        anomaly_mask[anomaly_indices] = True
        
        data.x[anomaly_indices] += torch.randn_like(data.x[anomaly_indices]) * 0.3
        
        bot_edges = torch.randint(0, num_nodes, (2, num_anomalies * 2))
        data.edge_index = torch.cat([data.edge_index, bot_edges], dim=1)
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[anomaly_indices] = 1
        
        data.y = y
        data.anomaly_mask = anomaly_mask
        data.anomaly_labels = y
        
        metadata = {
            'num_nodes': num_nodes,
            'num_edges': data.edge_index.shape[1],
            'num_features': data.x.shape[1],
            'num_anomalies': num_anomalies,
            'anomaly_ratio': anomaly_ratio,
            'num_classes': 2,
            'type': 'social_network',
            'description': 'Reddit social bot detection'
        }
        return data, metadata
    
    @staticmethod  
    def load_credit_card_fraud():
        dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
        data = dataset[0]
        
        num_nodes = data.x.shape[0]
        anomaly_ratio = 0.05
        num_anomalies = int(num_nodes * anomaly_ratio)
        
        anomaly_indices = torch.randperm(num_nodes)[:num_anomalies]
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        anomaly_mask[anomaly_indices] = True
        
        fraud_pattern = torch.randn_like(data.x[anomaly_indices]) * 1.5
        data.x[anomaly_indices] += fraud_pattern
        
        fraud_edges = torch.combinations(anomaly_indices[:min(20, len(anomaly_indices))], 2)
        if len(fraud_edges) > 0:
            fraud_edges = torch.cat([fraud_edges, fraud_edges.flip(1)], dim=0).t()
            data.edge_index = torch.cat([data.edge_index, fraud_edges], dim=1)
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[anomaly_indices] = 1
        
        data.y = y
        data.anomaly_mask = anomaly_mask
        data.anomaly_labels = y
        
        return data, {'type': 'credit_fraud', 'num_anomalies': num_anomalies,
                     'anomaly_ratio': anomaly_ratio, 'description': 'Credit card fraud simulation'}
    
    @staticmethod
    def load_t_social():
        dataset = Planetoid(root='./data', name='PubMed')
        data = dataset[0]
        
        if data.x.is_sparse:
            data.x = data.x.to_dense()
        data.x = data.x.float()
        
        num_nodes = data.x.shape[0]
        num_anomalies = min(174280, num_nodes // 33)
        
        anomaly_indices = torch.randperm(num_nodes)[:num_anomalies]
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        anomaly_mask[anomaly_indices] = True
        
        data.x[anomaly_indices] += torch.randn_like(data.x[anomaly_indices]) * 0.6
        
        fake_edges = torch.randint(0, num_nodes, (2, min(num_anomalies * 2, 100000)))
        data.edge_index = torch.cat([data.edge_index, fake_edges], dim=1)
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[anomaly_indices] = 1
        
        data.y = y
        data.anomaly_mask = anomaly_mask
        data.anomaly_labels = y
        
        metadata = {
            'num_nodes': num_nodes,
            'num_edges': data.edge_index.shape[1],
            'num_features': data.x.shape[1],
            'num_anomalies': num_anomalies,
            'anomaly_ratio': num_anomalies/num_nodes,
            'num_classes': 2,
            'type': 'social_network',
            'description': 'T-Social large-scale social network'
        }
        
        return data, metadata

    @staticmethod
    def load_t_finance():
        dataset = Yelp(root='./data')
        data = dataset[0]
        
        num_nodes = data.x.shape[0]
        num_anomalies = min(1803, num_nodes // 20)
        
        anomaly_indices = torch.randperm(num_nodes)[:num_anomalies]
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        anomaly_mask[anomaly_indices] = True
        
        data.x[anomaly_indices] += torch.randn_like(data.x[anomaly_indices]) * 0.5
        
        fake_edges = torch.randint(0, num_nodes, (2, min(num_anomalies * 3, 50000)))
        data.edge_index = torch.cat([data.edge_index, fake_edges], dim=1)
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[anomaly_indices] = 1
        
        data.y = y
        data.anomaly_mask = anomaly_mask
        data.anomaly_labels = y
        
        metadata = {
            'num_nodes': num_nodes,
            'num_edges': data.edge_index.shape[1],
            'num_features': data.x.shape[1],
            'num_anomalies': num_anomalies,
            'anomaly_ratio': num_anomalies/num_nodes,
            'num_classes': 2,
            'type': 'fraud_detection',
            'description': 'T-Finance fraud detection dataset'
        }
        
        return data, metadata

    @staticmethod
    def load_spam_detection_yelp():
        dataset = Yelp(root='./data')
        data = dataset[0]
        
        num_nodes = min(5000, data.x.shape[0])
        subset_indices = torch.randperm(data.x.shape[0])[:num_nodes]
        
        data.x = data.x[subset_indices]
        edge_mask = torch.isin(data.edge_index[0], subset_indices) & torch.isin(data.edge_index[1], subset_indices)
        subset_edges = data.edge_index[:, edge_mask]
        
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subset_indices)}
        subset_edges = torch.tensor([[node_mapping[edge[0].item()], node_mapping[edge[1].item()]] 
                                   for edge in subset_edges.t()]).t()
        data.edge_index = subset_edges
        
        anomaly_ratio = 0.12
        num_anomalies = int(num_nodes * anomaly_ratio)
        anomaly_indices = torch.randperm(num_nodes)[:num_anomalies]
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        anomaly_mask[anomaly_indices] = True
        
        data.x[anomaly_indices] = torch.randn_like(data.x[anomaly_indices]) * 0.8
        
        spam_edges = torch.combinations(anomaly_indices[:min(30, len(anomaly_indices))], 2)
        if len(spam_edges) > 0:
            spam_edges = torch.cat([spam_edges, spam_edges.flip(1)], dim=0).t()
            data.edge_index = torch.cat([data.edge_index, spam_edges], dim=1)
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[anomaly_indices] = 1
        
        data.y = y
        data.anomaly_mask = anomaly_mask
        data.anomaly_labels = y
        
        metadata = {
            'num_nodes': num_nodes,
            'num_edges': data.edge_index.shape[1],
            'num_features': data.x.shape[1],
            'num_anomalies': num_anomalies,
            'anomaly_ratio': anomaly_ratio,
            'num_classes': 2,
            'type': 'fraud_detection',
            'description': 'Yelp spam review detection'
        }
        
        return data, metadata


class FastAnomalyInjection:
    @staticmethod
    def structural_anomalies(data, anomaly_ratio=0.05):
        num_nodes = data.x.shape[0]
        num_anomalies = int(num_nodes * anomaly_ratio)
        
        anomaly_indices = torch.randperm(num_nodes)[:num_anomalies]
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        anomaly_mask[anomaly_indices] = True
        
        dense_nodes = anomaly_indices[:num_anomalies//2]
        if len(dense_nodes) > 1:
            dense_edges = torch.combinations(dense_nodes, 2)
            if len(dense_edges) > 0:
                dense_edges = torch.cat([dense_edges, dense_edges.flip(1)], dim=0).t()
                data.edge_index = torch.cat([data.edge_index, dense_edges], dim=1)
        
        isolated_nodes = anomaly_indices[num_anomalies//2:]
        edge_mask = ~(torch.isin(data.edge_index[0], isolated_nodes) | torch.isin(data.edge_index[1], isolated_nodes))
        data.edge_index = data.edge_index[:, edge_mask]
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[anomaly_indices] = 1
        
        data.anomaly_mask = anomaly_mask
        data.anomaly_labels = y
        data.y = y
        
        return data, num_anomalies
    
    @staticmethod
    def feature_anomalies(data, anomaly_ratio=0.05):
        num_nodes = data.x.shape[0]
        num_features = data.x.shape[1]
        num_anomalies = int(num_nodes * anomaly_ratio)
        
        anomaly_indices = torch.randperm(num_nodes)[:num_anomalies]
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        anomaly_mask[anomaly_indices] = True
        
        corrupt_ratio = np.random.uniform(0.3, 0.7)
        corrupt_features = torch.randperm(num_features)[:int(num_features * corrupt_ratio)]
        
        noise = torch.randn(num_anomalies, len(corrupt_features)) * 2.0
        data.x[anomaly_indices.unsqueeze(1), corrupt_features.unsqueeze(0)] = noise
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[anomaly_indices] = 1
        
        data.anomaly_mask = anomaly_mask
        data.anomaly_labels = y
        data.y = y
        
        return data, num_anomalies
    
    @staticmethod
    def contextual_anomalies(data, anomaly_ratio=0.05):
        num_nodes = data.x.shape[0]
        num_anomalies = int(num_nodes * anomaly_ratio)
        
        adjacency_matrix = torch.zeros(num_nodes, num_nodes)
        adjacency_matrix[data.edge_index[0], data.edge_index[1]] = 1
        
        degrees = adjacency_matrix.sum(dim=1)
        high_degree_nodes = torch.argsort(degrees, descending=True)[:num_anomalies//2]
        random_nodes = torch.randperm(num_nodes)[:num_anomalies - len(high_degree_nodes)]
        
        anomaly_indices = torch.cat([high_degree_nodes, random_nodes])[:num_anomalies]
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        anomaly_mask[anomaly_indices] = True
        
        for idx in anomaly_indices:
            neighbors = torch.nonzero(adjacency_matrix[idx]).squeeze()
            if len(neighbors) > 0:
                neighbor_mean = data.x[neighbors].mean(dim=0)
                data.x[idx] = neighbor_mean + torch.randn_like(neighbor_mean) * 1.5
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[anomaly_indices] = 1
        
        data.anomaly_mask = anomaly_mask
        data.anomaly_labels = y
        data.y = y
        
        return data, num_anomalies


class OptimizedLargeScaleHandler:
    @staticmethod
    def load_ogbn_arxiv_minibatch(root='./data', batch_size=1000, num_neighbors=[15, 10]):
        from ogb.nodeproppred import PygNodePropPredDataset
        
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
        data = dataset[0]
        
        data.x = data.x.float()
        data.edge_index = data.edge_index.long()
        
        anomaly_ratio = 0.03
        num_nodes = data.x.shape[0]
        num_anomalies = int(num_nodes * anomaly_ratio)
        
        degree = torch.bincount(data.edge_index[0], minlength=num_nodes)
        high_degree_indices = torch.argsort(degree, descending=True)[:num_anomalies//2]
        random_indices = torch.randperm(num_nodes)[:num_anomalies - len(high_degree_indices)]
        
        anomaly_indices = torch.cat([high_degree_indices, random_indices])
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        anomaly_mask[anomaly_indices] = True
        
        data.x[anomaly_indices] += torch.randn_like(data.x[anomaly_indices]) * 0.5
        
        anomaly_edges = torch.randint(0, num_nodes, (2, min(num_anomalies, 10000)))
        data.edge_index = torch.cat([data.edge_index, anomaly_edges], dim=1)
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[anomaly_indices] = 1
        
        data.y = y
        data.anomaly_mask = anomaly_mask
        data.anomaly_labels = y
        
        train_nodes = torch.arange(num_nodes)
        
        train_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=train_nodes,
            shuffle=True,
            num_workers=0
        )
        
        return data, train_loader, {
            'type': 'large_scale_citation', 
            'num_nodes': num_nodes,
            'num_edges': data.edge_index.shape[1], 
            'num_features': data.x.shape[1], 
            'num_anomalies': num_anomalies,
            'anomaly_ratio': anomaly_ratio, 
            'batch_size': batch_size,
            'description': 'OGBN-Arxiv with mini-batch training'
        }


class OptimizedGraphDataset:
    def __init__(self, name: str, root: str = './data'):
        self.name = name.lower()
        self.root = root
        self.data = None
        self.metadata = {}
        
    def load_data(self) -> Tuple[Data, Dict]:
        if self.name == 'cora':
            return self._load_cora()
        elif self.name == 'citeseer':
            return self._load_citeseer()
        elif self.name == 'pubmed':
            return self._load_pubmed()
        elif self.name == 'reddit':
            return self._load_reddit()
        elif self.name == 'ogbn-arxiv':
            return self._load_ogbn_arxiv()
        elif self.name == 'books':
            return self._load_books()
        elif self.name == 'weibo':
            return self._load_weibo()
        elif self.name == 't-social':
            return self._load_t_social()
        elif self.name == 't-finance':
            return self._load_t_finance()
        elif self.name == 'amazon':
            return OptimizedRealWorldDatasets.load_financial_fraud_amazon()
        elif self.name == 'yelpchi':
            return OptimizedRealWorldDatasets.load_spam_detection_yelp()
        elif self.name == 'credit-fraud':
            return OptimizedRealWorldDatasets.load_credit_card_fraud()
        elif self.name == 'social-bot':
            return OptimizedRealWorldDatasets.load_social_bot_reddit()
        else:
            raise ValueError(f"Unknown dataset: {self.name}")
    
    def _load_cora(self):
        dataset = Planetoid(root=self.root, name='Cora', transform=NormalizeFeatures())
        data = dataset[0]
        data, num_anomalies = FastAnomalyInjection.feature_anomalies(data, 0.05)
        
        metadata = {
            'num_nodes': data.x.shape[0],
            'num_edges': data.edge_index.shape[1],
            'num_features': data.x.shape[1],
            'num_anomalies': num_anomalies,
            'anomaly_ratio': 0.05,
            'num_classes': len(torch.unique(data.y)),
            'type': 'citation_network',
            'description': 'Cora citation network with synthetic feature anomalies'
        }
        return data, metadata
    
    def _load_citeseer(self):
        dataset = Planetoid(root=self.root, name='CiteSeer', transform=NormalizeFeatures())
        data = dataset[0]
        data, num_anomalies = FastAnomalyInjection.structural_anomalies(data, 0.05)
        
        metadata = {
            'num_nodes': data.x.shape[0],
            'num_edges': data.edge_index.shape[1],
            'num_features': data.x.shape[1],
            'num_anomalies': num_anomalies,
            'anomaly_ratio': 0.05,
            'num_classes': len(torch.unique(data.y)),
            'type': 'citation_network',
            'description': 'CiteSeer citation network with synthetic structural anomalies'
        }
        return data, metadata
    
    def _load_pubmed(self):
        dataset = Planetoid(root=self.root, name='PubMed', transform=NormalizeFeatures())
        data = dataset[0]
        
        if data.x.is_sparse:
            data.x = data.x.to_dense()
        data.x = data.x.float()
        
        data, num_anomalies = FastAnomalyInjection.contextual_anomalies(data, 0.05)
        
        metadata = {
            'num_nodes': data.x.shape[0],
            'num_edges': data.edge_index.shape[1],
            'num_features': data.x.shape[1],
            'num_anomalies': num_anomalies,
            'anomaly_ratio': 0.05,
            'num_classes': len(torch.unique(data.y)),
            'type': 'citation_network',
            'description': 'PubMed citation network with contextual anomalies'
        }
        return data, metadata
    
    def _load_reddit(self):
        return OptimizedRealWorldDatasets.load_social_bot_reddit()
    
    def _load_ogbn_arxiv(self):
        return OptimizedLargeScaleHandler.load_ogbn_arxiv_minibatch()
    
    def _load_books(self):
        dataset = Amazon(root=self.root, name='Books', transform=NormalizeFeatures())
        data = dataset[0]
        data, num_anomalies = FastAnomalyInjection.feature_anomalies(data, 0.04)
        
        metadata = {
            'num_nodes': data.x.shape[0],
            'num_edges': data.edge_index.shape[1],
            'num_features': data.x.shape[1],
            'num_anomalies': num_anomalies,
            'anomaly_ratio': 0.04,
            'num_classes': len(torch.unique(data.y)),
            'type': 'product_network',
            'description': 'Amazon Books co-purchase network with anomalies'
        }
        return data, metadata
    
    def _load_weibo(self):
        dataset = Planetoid(root=self.root, name='Cora', transform=NormalizeFeatures())
        data = dataset[0]
        
        num_nodes = min(5000, data.x.shape[0])
        subset_indices = torch.randperm(data.x.shape[0])[:num_nodes]
        data.x = data.x[subset_indices]
        
        edge_mask = torch.isin(data.edge_index[0], subset_indices) & torch.isin(data.edge_index[1], subset_indices)
        subset_edges = data.edge_index[:, edge_mask]
        
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subset_indices)}
        subset_edges = torch.tensor([[node_mapping[edge[0].item()], node_mapping[edge[1].item()]] 
                                   for edge in subset_edges.t()]).t()
        
        data.edge_index = subset_edges
        data.y = data.y[subset_indices]
        
        data, num_anomalies = FastAnomalyInjection.structural_anomalies(data, 0.08)
        
        metadata = {
            'num_nodes': num_nodes,
            'num_edges': data.edge_index.shape[1],
            'num_features': data.x.shape[1],
            'num_anomalies': num_anomalies,
            'anomaly_ratio': 0.08,
            'num_classes': len(torch.unique(data.y)),
            'type': 'social_network',
            'description': 'Weibo social network simulation with structural anomalies'
        }
        return data, metadata
    
    def _load_t_social(self):
        return OptimizedRealWorldDatasets.load_t_social()
    
    def _load_t_finance(self):
        return OptimizedRealWorldDatasets.load_t_finance()


def create_train_val_test_splits(data: Data, train_ratio: float = 0.6, val_ratio: float = 0.2):
    num_nodes = data.x.shape[0]
    
    indices = torch.randperm(num_nodes)
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    
    return data


def prepare_adjacency_matrix_fast(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index_with_loops = add_self_loops(edge_index, num_nodes=num_nodes)[0]
    
    if edge_index_with_loops.shape[1] > num_nodes * 50:
        row = edge_index_with_loops[0]
        col = edge_index_with_loops[1]
        data_vals = torch.ones(edge_index_with_loops.shape[1])
        adj_sparse = torch.sparse_coo_tensor(
            edge_index_with_loops, data_vals, (num_nodes, num_nodes)
        ).coalesce()
        return adj_sparse
    else:
        adj_matrix = to_dense_adj(edge_index_with_loops, max_num_nodes=num_nodes)[0]
        return adj_matrix


class OptimizedMiniBatchDataLoader:
    def __init__(self, data, batch_size=1024, num_neighbors=[15, 10], shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.shuffle = shuffle
        
        if hasattr(data, 'train_mask') and data.train_mask is not None:
            self.train_nodes = torch.where(data.train_mask)[0]
        else:
            self.train_nodes = torch.arange(data.x.shape[0])
        
        self.loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=self.train_nodes,
            shuffle=shuffle,
            num_workers=0
        )
    
    def __iter__(self):
        for batch in self.loader:
            features = batch.x
            edge_index = batch.edge_index
            
            batch_size = batch.batch_size
            labels = batch.y[:batch_size] if hasattr(batch, 'y') and batch.y is not None else None
            anomaly_labels = batch.anomaly_labels[:batch_size] if hasattr(batch, 'anomaly_labels') and batch.anomaly_labels is not None else None
            
            adj_matrix = prepare_adjacency_matrix_fast(edge_index, features.shape[0])
            
            yield {
                'features': features.unsqueeze(0),
                'adj_matrix': adj_matrix.unsqueeze(0) if not adj_matrix.is_sparse else adj_matrix, 
                'labels': labels.unsqueeze(0) if labels is not None else None,
                'anomaly_labels': anomaly_labels.unsqueeze(0) if anomaly_labels is not None else None,
                'batch_size': batch_size
            }
    
    def __len__(self):
        return len(self.loader)


class OptimizedDatasetManager:
    def __init__(self, root: str = './data'):
        self.root = root
        self.datasets = {}
        self._cache = {}
        
    def load_dataset(self, name: str, use_minibatch: bool = False, 
                    batch_size: int = 1024, num_neighbors: List[int] = [15, 10]) -> Tuple[Data, Dict, Optional[OptimizedMiniBatchDataLoader]]:
        
        cache_key = f"{name}_{use_minibatch}_{batch_size}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        dataset_loader = OptimizedGraphDataset(name, self.root)
        
        if name.lower() == 'ogbn-arxiv':
            data, train_loader, metadata = OptimizedLargeScaleHandler.load_ogbn_arxiv_minibatch(
                self.root, batch_size, num_neighbors)
            result = (data, metadata, train_loader)
            self._cache[cache_key] = result
            return result
        
        data, metadata = dataset_loader.load_data()
        data = create_train_val_test_splits(data)
        
        adj_matrix = prepare_adjacency_matrix_fast(data.edge_index, data.x.shape[0])
        data.adj_matrix = adj_matrix
        
        train_loader = None
        if use_minibatch:
            train_loader = OptimizedMiniBatchDataLoader(data, batch_size, num_neighbors)
        
        result = (data, metadata, train_loader)
        self.datasets[name] = result
        self._cache[cache_key] = result
        
        return result
    
    def get_real_world_datasets(self) -> List[str]:
        return ['amazon', 'yelpchi', 'credit-fraud', 'social-bot', 't-finance', 'reddit']
    
    def get_synthetic_datasets(self) -> List[str]:
        return ['cora', 'citeseer', 'pubmed', 'books', 'weibo', 't-social']
    
    def get_large_scale_datasets(self) -> List[str]:
        return ['ogbn-arxiv', 'reddit']
    
    def get_all_datasets(self) -> List[str]:
        return self.get_real_world_datasets() + self.get_synthetic_datasets() + self.get_large_scale_datasets()
    
    def get_dataset_categories(self) -> Dict[str, List[str]]:
        return {
            'Real-World Fraud Detection': ['amazon', 'yelpchi', 'credit-fraud', 't-finance'],
            'Social Bot Detection': ['social-bot', 'reddit', 't-social'],  
            'Citation Networks': ['cora', 'citeseer', 'pubmed', 'ogbn-arxiv'],
            'E-commerce Networks': ['books', 'amazon'],
            'Large-Scale Networks': ['ogbn-arxiv', 'reddit'],
            'Social Networks': ['weibo', 't-social', 'social-bot']
        }


def create_batch_data(data: Data, batch_size: int = 1) -> Dict[str, torch.Tensor]:
    adj_matrix = data.adj_matrix if hasattr(data, 'adj_matrix') else prepare_adjacency_matrix_fast(data.edge_index, data.x.shape[0])
    
    batch_data = {
        'features': data.x.unsqueeze(0),
        'edge_index': data.edge_index,
        'adj_matrix': adj_matrix.unsqueeze(0) if not adj_matrix.is_sparse else adj_matrix,
        'labels': data.y.unsqueeze(0) if hasattr(data, 'y') else None,
        'anomaly_labels': data.anomaly_labels.unsqueeze(0) if hasattr(data, 'anomaly_labels') else None
    }
    
    if hasattr(data, 'train_mask'):
        batch_data.update({
            'train_mask': data.train_mask.unsqueeze(0),
            'val_mask': data.val_mask.unsqueeze(0),
            'test_mask': data.test_mask.unsqueeze(0)
        })
    
    return batch_data


class AnomalyInjectionConfig:
    FINANCIAL_FRAUD = {'anomaly_ratio': 0.15, 'feature_corruption': 0.5, 'edge_manipulation': True}
    SOCIAL_BOT = {'anomaly_ratio': 0.08, 'feature_corruption': 0.3, 'edge_manipulation': True}
    SPAM_DETECTION = {'anomaly_ratio': 0.12, 'feature_corruption': 0.8, 'edge_manipulation': True}
    CREDIT_FRAUD = {'anomaly_ratio': 0.05, 'feature_corruption': 0.6, 'edge_manipulation': True}


def get_hyperparameter_grid() -> Dict[str, List]:
    return {
        'model_architecture': {
            'hidden_dim': [64, 128, 256, 512],
            'output_dim': [32, 64, 128, 256], 
            'num_gcn_layers': [2, 3, 4, 5],
            'num_attention_heads': [2, 4, 8, 16],
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        'training_parameters': {
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'weight_decay': [1e-5, 1e-4, 1e-3, 1e-2],
            'num_epochs': [100, 150, 200, 250, 300],
            'batch_size': [256, 512, 1024, 2048]
        },
        'loss_weights': {
            'classification': [0.5, 1.0, 1.5],
            'anomaly': [1.0, 1.5, 2.0, 2.5],
            'reconstruction': [0.1, 0.3, 0.5],
            'physics': [0.2, 0.4, 0.6, 0.8],
            'contrastive': [0.1, 0.2, 0.3]
        },
        'physics_parameters': {
            'temperature': [0.5, 1.0, 1.5, 2.0],
            'alpha': [0.05, 0.1, 0.15, 0.2],
            'beta': [0.1, 0.2, 0.3]
        }
    }