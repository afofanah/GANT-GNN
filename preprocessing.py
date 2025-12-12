import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, add_self_loops, degree
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from datetime import datetime


class DataPreprocessingPipeline:
    def __init__(self, seed=42, output_dir='./preprocessing_logs'):
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessing_log = []
        
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def log_step(self, step_name: str, details: Dict):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'details': details
        }
        self.preprocessing_log.append(log_entry)
    
    def normalize_features(self, data: Data, method='standard') -> Data:
        original_shape = data.x.shape
        
        if hasattr(data.x, 'is_sparse') and data.x.is_sparse:
            data.x = data.x.to_dense()
        
        data.x = data.x.float()
        
        if method == 'standard':
            scaler = StandardScaler()
            data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float32)
        elif method == 'minmax':
            scaler = MinMaxScaler()
            data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float32)
        elif method == 'l2':
            data.x = F.normalize(data.x, p=2, dim=1)
        
        self.log_step('feature_normalization', {
            'method': method,
            'original_shape': original_shape,
            'final_shape': data.x.shape,
            'feature_stats': {
                'mean': data.x.mean().item(),
                'std': data.x.std().item(),
                'min': data.x.min().item(),
                'max': data.x.max().item()
            }
        })
        
        return data
    
    def normalize_adjacency(self, edge_index: torch.Tensor, num_nodes: int, add_self_loops_flag=True) -> torch.Tensor:
        if add_self_loops_flag:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        
        row_sum = adj.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1
        adj_normalized = adj / row_sum
        
        self.log_step('adjacency_normalization', {
            'num_nodes': num_nodes,
            'num_edges': edge_index.shape[1],
            'self_loops_added': add_self_loops_flag,
            'density': (adj > 0).float().mean().item()
        })
        
        return adj_normalized
    
    def create_train_val_test_splits(self, data: Data, train_ratio=0.6, val_ratio=0.2, stratify_by_anomaly=True) -> Data:
        num_nodes = data.x.shape[0]
        indices = torch.arange(num_nodes)
        
        if stratify_by_anomaly and hasattr(data, 'anomaly_labels'):
            anomaly_labels = data.anomaly_labels.numpy()
            train_idx, temp_idx = train_test_split(
                indices.numpy(), 
                train_size=train_ratio, 
                stratify=anomaly_labels, 
                random_state=self.seed
            )
            
            temp_anomaly = anomaly_labels[temp_idx]
            val_size = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
            val_idx, test_idx = train_test_split(
                temp_idx, 
                train_size=val_size, 
                stratify=temp_anomaly, 
                random_state=self.seed
            )
            
            train_idx = torch.tensor(train_idx)
            val_idx = torch.tensor(val_idx)
            test_idx = torch.tensor(test_idx)
        else:
            shuffled_indices = indices[torch.randperm(num_nodes)]
            train_size = int(train_ratio * num_nodes)
            val_size = int(val_ratio * num_nodes)
            
            train_idx = shuffled_indices[:train_size]
            val_idx = shuffled_indices[train_size:train_size + val_size]
            test_idx = shuffled_indices[train_size + val_size:]
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        self.log_step('data_splitting', {
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': 1 - train_ratio - val_ratio,
            'stratified': stratify_by_anomaly and hasattr(data, 'anomaly_labels'),
            'train_anomaly_ratio': data.anomaly_labels[train_idx].float().mean().item() if hasattr(data, 'anomaly_labels') else None,
            'val_anomaly_ratio': data.anomaly_labels[val_idx].float().mean().item() if hasattr(data, 'anomaly_labels') else None,
            'test_anomaly_ratio': data.anomaly_labels[test_idx].float().mean().item() if hasattr(data, 'anomaly_labels') else None
        })
        
        return data
    
    def setup_minibatch_training(self, data: Data, batch_size=1024, num_neighbors=[15, 10]) -> NeighborLoader:
        if hasattr(data, 'train_mask') and data.train_mask is not None:
            train_nodes = torch.where(data.train_mask)[0]
        else:
            train_nodes = torch.arange(data.x.shape[0])
        
        loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=train_nodes,
            shuffle=True,
            num_workers=0
        )
        
        self.log_step('minibatch_setup', {
            'batch_size': batch_size,
            'num_neighbors': num_neighbors,
            'total_train_nodes': len(train_nodes),
            'estimated_batches': len(train_nodes) // batch_size + 1
        })
        
        return loader
    
    def save_preprocessing_log(self, filename=None):
        if filename is None:
            filename = f"preprocessing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        log_path = self.output_dir / filename
        
        with open(log_path, 'w') as f:
            json.dump(self.preprocessing_log, f, indent=2, default=str)
        
        return str(log_path)


class SyntheticAnomalyInjector:
    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def inject_structural_anomalies(self, data: Data, anomaly_ratio=0.05, injection_type='mixed') -> Tuple[Data, Dict]:
        num_nodes = data.x.shape[0]
        num_anomalies = int(num_nodes * anomaly_ratio)
        
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        if injection_type == 'dense_subgraph':
            anomaly_indices = torch.randperm(num_nodes)[:num_anomalies]
            anomaly_mask[anomaly_indices] = True
            
            dense_connections = torch.combinations(anomaly_indices, 2)
            if len(dense_connections) > 0:
                dense_edges = torch.cat([dense_connections, dense_connections.flip(1)], dim=0).t()
                data.edge_index = torch.cat([data.edge_index, dense_edges], dim=1)
        
        elif injection_type == 'isolated_nodes':
            anomaly_indices = torch.randperm(num_nodes)[:num_anomalies]
            anomaly_mask[anomaly_indices] = True
            
            edge_mask = ~(torch.isin(data.edge_index[0], anomaly_indices) | 
                         torch.isin(data.edge_index[1], anomaly_indices))
            data.edge_index = data.edge_index[:, edge_mask]
        
        elif injection_type == 'mixed':
            half_anomalies = num_anomalies // 2
            
            dense_indices = torch.randperm(num_nodes)[:half_anomalies]
            isolated_indices = torch.randperm(num_nodes)[half_anomalies:num_anomalies]
            
            anomaly_indices = torch.cat([dense_indices, isolated_indices])
            anomaly_mask[anomaly_indices] = True
            
            dense_connections = torch.combinations(dense_indices, 2)
            if len(dense_connections) > 0:
                dense_edges = torch.cat([dense_connections, dense_connections.flip(1)], dim=0).t()
                data.edge_index = torch.cat([data.edge_index, dense_edges], dim=1)
            
            edge_mask = ~(torch.isin(data.edge_index[0], isolated_indices) | 
                         torch.isin(data.edge_index[1], isolated_indices))
            data.edge_index = data.edge_index[:, edge_mask]
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[anomaly_indices] = 1
        
        data.y = y
        data.anomaly_mask = anomaly_mask
        data.anomaly_labels = y
        
        injection_details = {
            'injection_type': injection_type,
            'num_anomalies': num_anomalies,
            'anomaly_ratio': anomaly_ratio,
            'anomaly_indices': anomaly_indices.tolist(),
            'final_edges': data.edge_index.shape[1]
        }
        
        return data, injection_details
    
    def inject_feature_anomalies(self, data: Data, anomaly_ratio=0.05, corruption_methods=['gaussian_noise', 'feature_swap']) -> Tuple[Data, Dict]:
        num_nodes = data.x.shape[0]
        num_features = data.x.shape[1]
        num_anomalies = int(num_nodes * anomaly_ratio)
        
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        anomaly_indices = torch.randperm(num_nodes)[:num_anomalies]
        anomaly_mask[anomaly_indices] = True
        
        corruption_details = []
        
        for idx in anomaly_indices:
            method = np.random.choice(corruption_methods)
            
            if method == 'gaussian_noise':
                noise_ratio = np.random.uniform(0.3, 0.8)
                corrupt_features = torch.randperm(num_features)[:int(num_features * noise_ratio)]
                noise = torch.randn(len(corrupt_features)) * data.x[idx, corrupt_features].std() * 2
                data.x[idx, corrupt_features] += noise
                
                corruption_details.append({
                    'node': idx.item(),
                    'method': 'gaussian_noise',
                    'corrupted_features': corrupt_features.tolist(),
                    'noise_ratio': noise_ratio
                })
            
            elif method == 'feature_swap':
                swap_ratio = np.random.uniform(0.2, 0.6)
                swap_features = torch.randperm(num_features)[:int(num_features * swap_ratio)]
                partner_idx = torch.randint(0, num_nodes, (1,)).item()
                
                original_values = data.x[idx, swap_features].clone()
                data.x[idx, swap_features] = data.x[partner_idx, swap_features]
                
                corruption_details.append({
                    'node': idx.item(),
                    'method': 'feature_swap',
                    'swapped_features': swap_features.tolist(),
                    'partner_node': partner_idx,
                    'swap_ratio': swap_ratio
                })
            
            elif method == 'outlier_injection':
                outlier_ratio = np.random.uniform(0.1, 0.4)
                outlier_features = torch.randperm(num_features)[:int(num_features * outlier_ratio)]
                
                feature_stats = data.x[:, outlier_features]
                outlier_values = torch.randn_like(feature_stats[idx]) * 5 + feature_stats.mean(dim=0)
                data.x[idx, outlier_features] = outlier_values
                
                corruption_details.append({
                    'node': idx.item(),
                    'method': 'outlier_injection',
                    'outlier_features': outlier_features.tolist(),
                    'outlier_ratio': outlier_ratio
                })
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[anomaly_indices] = 1
        
        data.y = y
        data.anomaly_mask = anomaly_mask
        data.anomaly_labels = y
        
        injection_details = {
            'num_anomalies': num_anomalies,
            'anomaly_ratio': anomaly_ratio,
            'corruption_methods': corruption_methods,
            'corruption_details': corruption_details
        }
        
        return data, injection_details
    
    def inject_contextual_anomalies(self, data: Data, anomaly_ratio=0.05, community_based=True) -> Tuple[Data, Dict]:
        num_nodes = data.x.shape[0]
        num_anomalies = int(num_nodes * anomaly_ratio)
        
        if community_based:
            G = nx.from_edgelist(data.edge_index.t().numpy())
            communities = list(nx.community.greedy_modularity_communities(G))
            
            anomaly_indices = []
            for community in communities[:min(3, len(communities))]:
                if len(community) > 5:
                    community_list = list(community)
                    selected = np.random.choice(community_list, min(num_anomalies//3, len(community_list)//3), replace=False)
                    anomaly_indices.extend(selected)
            
            remaining = num_anomalies - len(anomaly_indices)
            if remaining > 0:
                all_nodes = set(range(num_nodes))
                used_nodes = set(anomaly_indices)
                available = list(all_nodes - used_nodes)
                additional = np.random.choice(available, min(remaining, len(available)), replace=False)
                anomaly_indices.extend(additional)
            
            anomaly_indices = torch.tensor(anomaly_indices[:num_anomalies])
        else:
            anomaly_indices = torch.randperm(num_nodes)[:num_anomalies]
        
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        anomaly_mask[anomaly_indices] = True
        
        for idx in anomaly_indices:
            neighbors = data.edge_index[1][data.edge_index[0] == idx]
            if len(neighbors) > 0:
                neighbor_mean = data.x[neighbors].mean(dim=0)
                neighborhood_std = data.x[neighbors].std(dim=0)
                
                contextual_anomaly = neighbor_mean + torch.randn_like(neighbor_mean) * neighborhood_std * 3
                data.x[idx] = contextual_anomaly
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[anomaly_indices] = 1
        
        data.y = y
        data.anomaly_mask = anomaly_mask
        data.anomaly_labels = y
        
        injection_details = {
            'num_anomalies': num_anomalies,
            'anomaly_ratio': anomaly_ratio,
            'community_based': community_based,
            'num_communities_used': len(communities) if community_based else 0,
            'anomaly_indices': anomaly_indices.tolist()
        }
        
        return data, injection_details


class RealWorldAnomalyPreprocessor:
    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def preprocess_financial_fraud(self, data: Data, fraud_patterns=['transaction_volume', 'network_behavior', 'temporal_patterns']) -> Tuple[Data, Dict]:
        num_nodes = data.x.shape[0]
        
        degree_centrality = degree(data.edge_index[0], num_nodes=num_nodes).float()
        high_degree_threshold = torch.quantile(degree_centrality, 0.9)
        potential_fraudsters = torch.where(degree_centrality > high_degree_threshold)[0]
        
        fraud_ratio = 0.15
        num_fraudsters = int(num_nodes * fraud_ratio)
        selected_fraudsters = potential_fraudsters[:min(len(potential_fraudsters), num_fraudsters//2)]
        
        remaining_fraudsters = num_fraudsters - len(selected_fraudsters)
        if remaining_fraudsters > 0:
            remaining_nodes = torch.randperm(num_nodes)[:remaining_fraudsters]
            selected_fraudsters = torch.cat([selected_fraudsters, remaining_nodes])
        
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        anomaly_mask[selected_fraudsters] = True
        
        fraud_details = []
        
        for fraud_idx in selected_fraudsters:
            pattern = np.random.choice(fraud_patterns)
            
            if pattern == 'transaction_volume':
                volume_multiplier = np.random.uniform(5.0, 15.0)
                volume_features = torch.randperm(data.x.shape[1])[:data.x.shape[1]//3]
                data.x[fraud_idx, volume_features] *= volume_multiplier
                
            elif pattern == 'network_behavior':
                fraud_connections = torch.randint(0, num_nodes, (np.random.randint(10, 30),))
                for target in fraud_connections:
                    new_edge = torch.tensor([[fraud_idx.item(), target.item()], [target.item(), fraud_idx.item()]], dtype=torch.long).t()
                    data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
            
            elif pattern == 'temporal_patterns':
                temporal_features = torch.randperm(data.x.shape[1])[:data.x.shape[1]//4]
                burst_pattern = torch.randn_like(data.x[fraud_idx, temporal_features]) * 3
                data.x[fraud_idx, temporal_features] += burst_pattern
            
            fraud_details.append({
                'fraud_node': fraud_idx.item(),
                'pattern': pattern,
                'degree_centrality': degree_centrality[fraud_idx].item()
            })
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[selected_fraudsters] = 1
        
        data.y = y
        data.anomaly_mask = anomaly_mask
        data.anomaly_labels = y
        
        preprocessing_details = {
            'fraud_type': 'financial_fraud',
            'num_fraudsters': len(selected_fraudsters),
            'fraud_ratio': fraud_ratio,
            'patterns_used': fraud_patterns,
            'fraud_details': fraud_details
        }
        
        return data, preprocessing_details
    
    def preprocess_social_bots(self, data: Data, bot_behaviors=['high_activity', 'coordinated_behavior', 'content_similarity']) -> Tuple[Data, Dict]:
        num_nodes = data.x.shape[0]
        
        activity_scores = degree(data.edge_index[0], num_nodes=num_nodes).float()
        high_activity_threshold = torch.quantile(activity_scores, 0.85)
        potential_bots = torch.where(activity_scores > high_activity_threshold)[0]
        
        bot_ratio = 0.08
        num_bots = int(num_nodes * bot_ratio)
        selected_bots = potential_bots[:min(len(potential_bots), num_bots)]
        
        if len(selected_bots) < num_bots:
            remaining = num_bots - len(selected_bots)
            additional_bots = torch.randperm(num_nodes)[:remaining]
            selected_bots = torch.cat([selected_bots, additional_bots])
        
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        anomaly_mask[selected_bots] = True
        
        bot_details = []
        
        for i, bot_idx in enumerate(selected_bots):
            behavior = np.random.choice(bot_behaviors)
            
            if behavior == 'high_activity':
                activity_multiplier = np.random.uniform(3.0, 8.0)
                activity_features = torch.randperm(data.x.shape[1])[:data.x.shape[1]//2]
                data.x[bot_idx, activity_features] *= activity_multiplier
                
            elif behavior == 'coordinated_behavior':
                if i < len(selected_bots) - 1:
                    coordination_partner = selected_bots[i + 1]
                    coordination_features = torch.randperm(data.x.shape[1])[:data.x.shape[1]//3]
                    avg_features = (data.x[bot_idx, coordination_features] + data.x[coordination_partner, coordination_features]) / 2
                    data.x[bot_idx, coordination_features] = avg_features
                    data.x[coordination_partner, coordination_features] = avg_features
                    
            elif behavior == 'content_similarity':
                template_features = torch.randn(data.x.shape[1]) * 0.5
                similarity_features = torch.randperm(data.x.shape[1])[:data.x.shape[1]//2]
                data.x[bot_idx, similarity_features] = template_features[similarity_features]
            
            bot_details.append({
                'bot_node': bot_idx.item(),
                'behavior': behavior,
                'activity_score': activity_scores[bot_idx].item()
            })
        
        bot_ring_edges = []
        for i in range(len(selected_bots)):
            for j in range(i+1, min(i+4, len(selected_bots))):
                bot_ring_edges.extend([
                    [selected_bots[i].item(), selected_bots[j].item()],
                    [selected_bots[j].item(), selected_bots[i].item()]
                ])
        
        if bot_ring_edges:
            bot_edges = torch.tensor(bot_ring_edges, dtype=torch.long).t()
            data.edge_index = torch.cat([data.edge_index, bot_edges], dim=1)
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[selected_bots] = 1
        
        data.y = y
        data.anomaly_mask = anomaly_mask
        data.anomaly_labels = y
        
        preprocessing_details = {
            'anomaly_type': 'social_bots',
            'num_bots': len(selected_bots),
            'bot_ratio': bot_ratio,
            'behaviors_used': bot_behaviors,
            'bot_details': bot_details
        }
        
        return data, preprocessing_details


class HyperparameterGrid:
    @staticmethod
    def get_complete_grid() -> Dict[str, Dict[str, List]]:
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
            },
            'data_preprocessing': {
                'feature_normalization': ['standard', 'minmax', 'l2'],
                'anomaly_injection_ratio': [0.03, 0.05, 0.07, 0.1],
                'train_val_test_split': [(0.6, 0.2, 0.2), (0.7, 0.15, 0.15), (0.8, 0.1, 0.1)]
            }
        }
    
    @staticmethod
    def get_optimized_configs() -> Dict[str, Dict]:
        return {
            'financial_fraud': {
                'hidden_dim': 256,
                'output_dim': 128,
                'num_gcn_layers': 4,
                'num_attention_heads': 8,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'anomaly_weight': 2.0,
                'physics_weight': 0.6
            },
            'social_bot_detection': {
                'hidden_dim': 128,
                'output_dim': 64,
                'num_gcn_layers': 3,
                'num_attention_heads': 4,
                'dropout': 0.2,
                'learning_rate': 0.005,
                'weight_decay': 1e-3,
                'anomaly_weight': 1.5,
                'physics_weight': 0.4
            },
            'citation_networks': {
                'hidden_dim': 128,
                'output_dim': 64,
                'num_gcn_layers': 3,
                'num_attention_heads': 4,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'anomaly_weight': 1.0,
                'physics_weight': 0.4
            },
            'large_scale': {
                'hidden_dim': 256,
                'output_dim': 128,
                'num_gcn_layers': 3,
                'num_attention_heads': 4,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'batch_size': 1024,
                'num_neighbors': [15, 10],
                'anomaly_weight': 1.5,
                'physics_weight': 0.5
            }
        }


class MemoryOptimizedPreprocessor:
    def __init__(self, max_memory_gb=8):
        self.max_memory_gb = max_memory_gb
    
    def check_memory_requirements(self, num_nodes: int, num_features: int, batch_size: int = None) -> Dict:
        node_features_gb = (num_nodes * num_features * 4) / (1024**3)
        adjacency_dense_gb = (num_nodes * num_nodes * 4) / (1024**3)
        
        if batch_size:
            batch_features_gb = (batch_size * num_features * 4) / (1024**3)
            batch_adjacency_gb = (batch_size * batch_size * 4) / (1024**3)
        else:
            batch_features_gb = node_features_gb
            batch_adjacency_gb = adjacency_dense_gb
        
        memory_estimate = {
            'node_features_gb': node_features_gb,
            'adjacency_dense_gb': adjacency_dense_gb,
            'batch_features_gb': batch_features_gb,
            'batch_adjacency_gb': batch_adjacency_gb,
            'total_estimate_gb': node_features_gb + batch_adjacency_gb,
            'requires_minibatch': adjacency_dense_gb > self.max_memory_gb,
            'recommended_batch_size': None
        }
        
        if memory_estimate['requires_minibatch'] and not batch_size:
            target_adjacency_gb = self.max_memory_gb * 0.5
            recommended_batch = int(np.sqrt(target_adjacency_gb * (1024**3) / 4))
            memory_estimate['recommended_batch_size'] = min(recommended_batch, 2048)
        
        return memory_estimate
    
    def setup_memory_efficient_training(self, data: Data, target_memory_gb: float = 4.0) -> Tuple[Optional[NeighborLoader], Dict]:
        memory_check = self.check_memory_requirements(
            data.x.shape[0], data.x.shape[1]
        )
        
        if memory_check['requires_minibatch']:
            batch_size = memory_check['recommended_batch_size']
            
            loader = NeighborLoader(
                data,
                num_neighbors=[15, 10],
                batch_size=batch_size,
                input_nodes=torch.arange(data.x.shape[0]),
                shuffle=True,
                num_workers=0
            )
            
            optimization_details = {
                'memory_optimization': True,
                'batch_size': batch_size,
                'num_neighbors': [15, 10],
                'estimated_memory_gb': memory_check['total_estimate_gb'],
                'loader_provided': True
            }
            
            return loader, optimization_details
        else:
            optimization_details = {
                'memory_optimization': False,
                'full_batch_possible': True,
                'estimated_memory_gb': memory_check['total_estimate_gb'],
                'loader_provided': False
            }
            
            return None, optimization_details


def preprocessing_report(preprocessing_steps: List[Dict], output_path: str):
    report = {
        'preprocessing_summary': {
            'total_steps': len(preprocessing_steps),
            'preprocessing_pipeline': [step['step'] for step in preprocessing_steps],
            'generated_at': datetime.now().isoformat()
        },
        'detailed_steps': preprocessing_steps,
        'reproducibility_info': {
            'random_seed': 42,
            'torch_version': torch.__version__,
            'numpy_version': np.__version__
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Preprocessing report saved to: {output_path}")
    return report