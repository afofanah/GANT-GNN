import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import os
import random
import logging
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch_geometric.datasets import Planetoid,  Reddit2
from ogb.nodeproppred import PygNodePropPredDataset
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from models.model_v1 import (
    GANTGNN, 
    EarlyStopping,
    LearningRateScheduler,
    AdaptiveThresholdSelector
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set matplotlib parameters for consistent, clean plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0
})


class GraphDataset(Dataset):
    """Enhanced dataset class with preprocessing capabilities"""
    
    def __init__(self, features, adj_matrix, labels, anomaly_labels=None, normalize=True):
        # Validate input dimensions
        if features.shape[0] != adj_matrix.shape[0]:
            raise ValueError(f"Feature matrix and adjacency matrix size mismatch: "
                           f"features {features.shape[0]} vs adjacency {adj_matrix.shape[0]}")
        
        # Preprocessing
        if normalize:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        
        self.features = torch.FloatTensor(features)
        self.adj_matrix = torch.FloatTensor(adj_matrix)
        self.labels = torch.LongTensor(labels)
        self.anomaly_labels = torch.FloatTensor(anomaly_labels) if anomaly_labels is not None else None
        self.adj_matrix = self._normalize_adjacency(self.adj_matrix)
        
    def _normalize_adjacency(self, adj):
        """Normalize adjacency matrix with self-loops"""
        adj_with_self_loops = adj + torch.eye(adj.size(0))
        degree = torch.sum(adj_with_self_loops, dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
        degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_normalized = torch.mm(torch.mm(degree_matrix_inv_sqrt, adj_with_self_loops), 
                                 degree_matrix_inv_sqrt)
        return adj_normalized
        
    def __len__(self):
        return 1  # Single graph
    
    def __getitem__(self, idx):
        return {
            'features': self.features.unsqueeze(0),
            'adj_matrix': self.adj_matrix.unsqueeze(0),
            'labels': self.labels,
            'anomaly_labels': self.anomaly_labels
        }


class DatasetLoader:
    """Enhanced dataset loader with better preprocessing and anomaly injection"""
    
    def __init__(self, root_dir='./data'):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)
        
    def load_citation_dataset(self, name):
        """Load citation datasets with enhanced preprocessing"""
        dataset = Planetoid(root=self.root_dir, name=name.capitalize())
        data = dataset[0]
        
        features = data.x.numpy()
        edge_index = data.edge_index.numpy()
        labels = data.y.numpy()
        num_nodes = features.shape[0]
        adj_matrix = np.zeros((num_nodes, num_nodes))
        adj_matrix[edge_index[0], edge_index[1]] = 1
        adj_matrix = adj_matrix + adj_matrix.T
        adj_matrix[adj_matrix > 1] = 1
        
        return features, adj_matrix, labels
    
    def load_ogbn_arxiv_dataset(self, max_nodes=10000):
        """Load OGBN-Arxiv dataset with sampling for efficiency"""
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=self.root_dir)
        data = dataset[0]
        
        num_nodes = min(max_nodes, data.x.size(0))
        indices = torch.randperm(data.x.size(0))[:num_nodes]
        
        features = data.x[indices].numpy()
        edge_index = data.edge_index.numpy()
        labels = data.y[indices].numpy().flatten()
        
        indices_set = set(indices.numpy())
        mask = np.isin(edge_index[0], indices.numpy()) & np.isin(edge_index[1], indices.numpy())
        edge_index = edge_index[:, mask]
        
        node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(indices)}
        edge_index[0] = [node_map[idx] for idx in edge_index[0]]
        edge_index[1] = [node_map[idx] for idx in edge_index[1]]
        edge_index = np.array(edge_index)
        
        adj_matrix = np.zeros((num_nodes, num_nodes))
        if edge_index.shape[1] > 0:
            adj_matrix[edge_index[0], edge_index[1]] = 1
            adj_matrix = adj_matrix + adj_matrix.T
            adj_matrix[adj_matrix > 1] = 1
        
        logger.info(f"OGBN-Arxiv loaded: {num_nodes} nodes, {features.shape[1]} features, {len(np.unique(labels))} classes")
        return features, adj_matrix, labels
    
    def load_reddit_dataset(self, max_nodes=5000):
        """Load Reddit dataset with sampling for efficiency"""
        dataset = Reddit2(root=self.root_dir)
        data = dataset[0]
        
        num_nodes = min(max_nodes, data.x.size(0))
        indices = torch.randperm(data.x.size(0))[:num_nodes]
        
        features = data.x[indices].numpy()
        edge_index = data.edge_index.numpy()
        labels = data.y[indices].numpy()
        
        indices_set = set(indices.numpy())
        mask = np.isin(edge_index[0], indices.numpy()) & np.isin(edge_index[1], indices.numpy())
        edge_index = edge_index[:, mask]
        
        node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(indices)}
        edge_index[0] = [node_map[idx] for idx in edge_index[0]]
        edge_index[1] = [node_map[idx] for idx in edge_index[1]]
        edge_index = np.array(edge_index)
        
        adj_matrix = np.zeros((num_nodes, num_nodes))
        if edge_index.shape[1] > 0:
            adj_matrix[edge_index[0], edge_index[1]] = 1
            adj_matrix = adj_matrix + adj_matrix.T
            adj_matrix[adj_matrix > 1] = 1
        
        return features, adj_matrix, labels
    
    def create_synthetic_dataset(self, dataset_type='books', anomaly_ratio=0.05):
        """Create synthetic datasets with realistic properties and consistent dimensions"""
        if dataset_type == 'books':
            base_num_nodes, num_features, num_classes = 1418, 21, 7
        elif dataset_type == 'weibo':
            base_num_nodes, num_features, num_classes = 8405, 400, 3
        else:
            base_num_nodes, num_features, num_classes = 2000, 32, 5
        
        community_size = base_num_nodes // num_classes
        num_nodes = community_size * num_classes
        
        logger.info(f"Creating synthetic dataset: {num_nodes} nodes ({num_classes} communities of {community_size})")
        features = np.random.randn(num_nodes, num_features)
        
        # Add community structure to features
        for i in range(num_classes):
            start_idx = i * community_size
            end_idx = (i + 1) * community_size
            features[start_idx:end_idx] += np.random.randn(num_features) * 0.5
        
        community_sizes = [community_size] * num_classes
        prob_matrix = [[0.8 if i == j else 0.1 for j in range(num_classes)] for i in range(num_classes)]
        
        G = nx.stochastic_block_model(community_sizes, prob_matrix)
        
        actual_num_nodes = len(G.nodes())
        num_random_edges = actual_num_nodes // 10
        
        existing_edges = set(G.edges())
        added_edges = 0
        attempts = 0
        max_attempts = num_random_edges * 5
        
        while added_edges < num_random_edges and attempts < max_attempts:
            u, v = random.sample(list(G.nodes()), 2)
            if (u, v) not in existing_edges and (v, u) not in existing_edges:
                G.add_edge(u, v)
                existing_edges.add((u, v))
                existing_edges.add((v, u))
                added_edges += 1
            attempts += 1
        
        adj_matrix = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).toarray()
        actual_nodes = adj_matrix.shape[0]
        
        if features.shape[0] != actual_nodes:
            if features.shape[0] > actual_nodes:
                features = features[:actual_nodes]
            else:
                padding = np.zeros((actual_nodes - features.shape[0], num_features))
                features = np.vstack([features, padding])
        
        labels = []
        for i in range(num_classes):
            labels.extend([i] * min(community_size, actual_nodes - len(labels)))
        
        labels = np.array(labels[:actual_nodes])
        if len(labels) < actual_nodes:
            labels = np.pad(labels, (0, actual_nodes - len(labels)), mode='edge')
        
        logger.info(f"Final dataset: features {features.shape}, adjacency {adj_matrix.shape}, labels {labels.shape}")
        return features, adj_matrix, labels
    
    def inject_enhanced_anomalies(self, features, adj_matrix, anomaly_ratio=0.05):
        """Inject diverse types of anomalies with proper bounds checking"""
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.from_numpy(adj_matrix).float()
        
        if features.shape[0] != adj_matrix.shape[0]:
            raise ValueError(f"Dimension mismatch: features {features.shape[0]} vs adjacency {adj_matrix.shape[0]}")
        
        num_nodes = adj_matrix.shape[0]
        num_anomalies = max(1, int(num_nodes * anomaly_ratio))
        
        logger.info(f"Injecting {num_anomalies} anomalies into {num_nodes} nodes")
        
        modified_features = features.clone()
        modified_adj = adj_matrix.clone()
        anomaly_labels = torch.zeros(num_nodes)
        
        anomaly_indices = np.random.choice(num_nodes, num_anomalies, replace=False)
        anomaly_labels[anomaly_indices] = 1
        
        for i, idx in enumerate(anomaly_indices):
            if idx >= num_nodes:
                continue
            
            anomaly_type = np.random.choice(['feature', 'structure', 'mixed'], p=[0.4, 0.4, 0.2])
            
            if anomaly_type in ['feature', 'mixed']:
                noise_type = np.random.choice(['gaussian', 'uniform', 'zero'])
                if noise_type == 'gaussian':
                    noise = torch.randn_like(modified_features[idx]) * 0.5
                    modified_features[idx] += noise
                elif noise_type == 'uniform':
                    noise = torch.rand_like(modified_features[idx]) - 0.5
                    modified_features[idx] += noise
                else:
                    modified_features[idx] = 0
            
            if anomaly_type in ['structure', 'mixed']:
                anomaly_subtype = np.random.choice(['isolation', 'hub', 'bridge'])
                
                if anomaly_subtype == 'isolation':
                    modified_adj[idx, :] = 0
                    modified_adj[:, idx] = 0
                
                elif anomaly_subtype == 'hub':
                    num_connections = min(20, num_nodes - 1)
                    valid_nodes = [i for i in range(num_nodes) if i != idx]
                    
                    if valid_nodes and num_connections > 0:
                        actual_connections = min(num_connections, len(valid_nodes))
                        random_nodes = np.random.choice(valid_nodes, actual_connections, replace=False)
                        
                        for node in random_nodes:
                            if 0 <= node < num_nodes and 0 <= idx < num_nodes:
                                modified_adj[idx, node] = 1
                                modified_adj[node, idx] = 1
                
                elif anomaly_subtype == 'bridge':
                    num_bridges = min(5, num_nodes - 1)
                    valid_nodes = [i for i in range(num_nodes) if i != idx]
                    
                    if valid_nodes and num_bridges > 0:
                        actual_bridges = min(num_bridges, len(valid_nodes))
                        bridge_nodes = np.random.choice(valid_nodes, actual_bridges, replace=False)
                        
                        for bridge_node in bridge_nodes:
                            if 0 <= bridge_node < num_nodes and 0 <= idx < num_nodes:
                                modified_adj[idx, bridge_node] = 1
                                modified_adj[bridge_node, idx] = 1
        
        assert modified_features.shape[0] == modified_adj.shape[0], "Final dimension mismatch after anomaly injection"
        return modified_features, modified_adj, anomaly_labels


class EnhancedTrainer:
    """trainer with comprehensive evaluation and integrated plotting"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', output_dir=None):
        self.device = device
        self.model = model.to(device)
        self.history = defaultdict(list)
        self.threshold_selector = AdaptiveThresholdSelector(method='f1_optimal')
        self.best_model_state = None
        self.best_f1 = 0
        self.best_auc = 0
        self.output_dir = Path(output_dir) if output_dir else Path('./Results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self, dataloader, optimizer, epoch):
        """training epoch with better loss tracking"""
        self.model.train()
        total_loss = 0
        loss_components = defaultdict(float)
        num_batches = 0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            adj_matrix = batch['adj_matrix'].to(self.device)
            labels = batch['labels'].to(self.device)
            anomaly_labels = batch.get('anomaly_labels')
            
            if anomaly_labels is not None:
                anomaly_labels = anomaly_labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(features, adj_matrix, labels, anomaly_labels)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if 'loss_components' in outputs:
                for key, value in outputs['loss_components'].items():
                    loss_components[key] += value
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        return avg_loss, avg_components
    
    def evaluate_comprehensive(self, dataloader, use_adaptive_threshold=True):
        """Comprehensive evaluation with multiple metrics"""
        self.model.eval()
        all_scores = []
        all_labels = []
        all_predictions = []
        all_class_logits = []
        all_class_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                adj_matrix = batch['adj_matrix'].to(self.device)
                labels = batch['labels'].to(self.device)
                anomaly_labels = batch.get('anomaly_labels')
                outputs = self.model(features, adj_matrix, labels, anomaly_labels)
                
                if anomaly_labels is not None:
                    results = self.model.detect_anomalies(
                        features, adj_matrix,
                        use_adaptive_threshold=use_adaptive_threshold,
                        validation_labels=anomaly_labels
                    )
                    
                    scores = results['fused_scores'].cpu().numpy().flatten()
                    predictions = results['predictions'].cpu().numpy().flatten()
                    true_labels = anomaly_labels.cpu().numpy().flatten()
                    
                    min_length = min(len(scores), len(true_labels), len(predictions))
                    all_scores.extend(scores[:min_length])
                    all_labels.extend(true_labels[:min_length])
                    all_predictions.extend(predictions[:min_length])
                
                class_logits = outputs['logits'].cpu().squeeze()
                if class_logits.dim() == 1:
                    class_logits = class_logits.unsqueeze(0)
                
                all_class_logits.append(class_logits)
                all_class_labels.append(labels.cpu())
        
        metrics = {}
        
        # Anomaly detection metrics
        if len(all_scores) > 0:
            all_scores = np.array(all_scores)
            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)
            
            auc_score = roc_auc_score(all_labels, all_scores)
            ap_score = average_precision_score(all_labels, all_scores)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='binary', zero_division=0
            )
            
            thresholds = np.linspace(0, 1, 100)
            f1_scores = []
            for threshold in thresholds:
                preds = (all_scores > threshold).astype(int)
                if len(np.unique(preds)) > 1:
                    f1_scores.append(roc_auc_score(all_labels, preds) if len(np.unique(all_labels)) > 1
                                   else precision_recall_fscore_support(all_labels, preds, average='binary')[2])
                else:
                    f1_scores.append(0)
            
            best_f1_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[best_f1_idx]
            optimal_predictions = (all_scores > optimal_threshold).astype(int)
            optimal_f1 = precision_recall_fscore_support(all_labels, optimal_predictions, average='binary')[2]
            
            metrics.update({
                'anomaly_auc': auc_score,
                'anomaly_ap': ap_score,
                'anomaly_precision': precision,
                'anomaly_recall': recall,
                'anomaly_f1': f1,
                'anomaly_optimal_f1': optimal_f1,
                'anomaly_optimal_threshold': optimal_threshold,
                'anomaly_predictions': all_predictions,
                'anomaly_scores': all_scores,
                'anomaly_true_labels': all_labels
            })
        
        # Classification metrics
        if len(all_class_logits) > 0:
            all_class_logits = torch.cat(all_class_logits, dim=0)
            all_class_labels = torch.cat(all_class_labels, dim=0)
            
            class_predictions = torch.argmax(all_class_logits, dim=1)
            class_accuracy = (class_predictions == all_class_labels).float().mean().item()
            
            metrics.update({
                'classification_accuracy': class_accuracy,
                'classification_predictions': class_predictions.numpy(),
                'classification_true_labels': all_class_labels.numpy()
            })
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs=200, lr=0.001, weight_decay=1e-4,
              patience=30, min_delta=0.001, scheduler_patience=10, dataset_name='experiment'):
        """Enhanced training loop with integrated plotting"""
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = LearningRateScheduler(optimizer, mode='plateau',
                                        patience=scheduler_patience, factor=0.5)
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        
        logger.info(f"Starting enhanced training for {num_epochs} epochs")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Store training data for plots
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset_name = dataset_name
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            train_loss, train_components = self.train_epoch(train_loader, optimizer, epoch)
            val_metrics = self.evaluate_comprehensive(val_loader, use_adaptive_threshold=True)
            epoch_time = time.time() - start_time
            
            self.history['train_loss'].append(train_loss)
            self.history['val_metrics'].append(val_metrics)
            self.history['loss_components'].append(train_components)
            
            val_f1 = val_metrics.get('anomaly_f1', 0)
            scheduler.step(val_f1)
            
            if early_stopping(val_f1, self.model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            val_auc = val_metrics.get('anomaly_auc', 0)
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_auc = val_auc
                self.best_model_state = self.model.state_dict().copy()
            
            if epoch % 10 == 0 or epoch < 5:
                logger.info(f"Epoch {epoch+1:3d}/{num_epochs}")
                logger.info(f"  Train Loss: {train_loss:.4f}")
                logger.info(f"  Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")
                logger.info(f"  Val Precision: {val_metrics.get('anomaly_precision', 0):.4f}")
                logger.info(f"  Val Recall: {val_metrics.get('anomaly_recall', 0):.4f}")
                logger.info(f"  Classification Acc: {val_metrics.get('classification_accuracy', 0):.4f}")
                logger.info(f"  Time: {epoch_time:.2f}s")
                
                if train_components:
                    comp_str = ", ".join([f"{k}: {v:.3f}" for k, v in train_components.items()])
                    logger.info(f"  Loss Components: {comp_str}")
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model - F1: {self.best_f1:.4f}, AUC: {self.best_auc:.4f}")
        self._generate_all_plots()
        
        return self.history
    
    def _generate_all_plots(self):
        """Generate all available plots and save to Results folder"""
        plots_dir = self.output_dir / 'plots' / self.dataset_name
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating comprehensive plots for {self.dataset_name}...")
        
        # Import plotting functions
        from utils_v1 import ResultAnalyzer
        analyzer = ResultAnalyzer()
        
        analyzer_results = {
            self.dataset_name: {
                'trainer': self,
                'model': self.model,
                'use_physics': getattr(self.model, 'use_physics', False)
            }
        }
        self._plot_training_curves(plots_dir / 'training_curves.pdf')
        self._plot_loss_components(plots_dir / 'loss_components.pdf')
        self._plot_roc_pr_curves(plots_dir / 'roc_pr_curves.pdf')
        self._plot_confusion_matrix(plots_dir / 'confusion_matrix.pdf')
        self._plot_error_analysis(plots_dir / 'error_analysis.pdf')
        analyzer.plot_embedding_analysis(analyzer_results, self.dataset_name, 
                                       plots_dir / 'embedding_analysis.pdf')
        
        if getattr(self.model, 'use_physics', False):
            analyzer.plot_physics_analysis(analyzer_results, self.dataset_name,
                                         plots_dir / 'physics_analysis.pdf')
        self._save_metrics_csv(plots_dir.parent / 'metrics')
        
        logger.info(f"All plots saved to: {plots_dir}")
        
    def _plot_training_curves(self, save_path):
        """Plot training curves with subplot labels"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Training Curves - {self.dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        positions = [(0,0), (0,1), (1,0), (1,1)]
        
        for label, pos in zip(subplot_labels, positions):
            axes[pos].text(-0.1, 1.05, label, transform=axes[pos].transAxes, 
                          fontsize=14, fontweight='bold', va='bottom', ha='right')

        axes[0, 0].plot(self.history['train_loss'], color='#2E86C1', linewidth=2)
        axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)

        epochs = range(len(self.history['val_metrics']))
        val_aucs = [m.get('anomaly_auc', 0) for m in self.history['val_metrics']]
        val_f1s = [m.get('anomaly_f1', 0) for m in self.history['val_metrics']]
        
        axes[0, 1].plot(epochs, val_aucs, label='AUC', marker='o', markersize=4, color='#E74C3C', linewidth=2)
        axes[0, 1].plot(epochs, val_f1s, label='F1', marker='s', markersize=4, color='#27AE60', linewidth=2)
        axes[0, 1].set_title('Validation Metrics', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Score', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision and Recall
        val_precisions = [m.get('anomaly_precision', 0) for m in self.history['val_metrics']]
        val_recalls = [m.get('anomaly_recall', 0) for m in self.history['val_metrics']]
        
        axes[1, 0].plot(epochs, val_precisions, label='Precision', marker='^', markersize=4, color='#8E44AD', linewidth=2)
        axes[1, 0].plot(epochs, val_recalls, label='Recall', marker='v', markersize=4, color='#F39C12', linewidth=2)
        axes[1, 0].set_title('Precision & Recall', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Score', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Classification accuracy
        val_accuracies = [m.get('classification_accuracy', 0) for m in self.history['val_metrics']]
        axes[1, 1].plot(epochs, val_accuracies, label='Accuracy', marker='d', markersize=4, color='#16A085', linewidth=2)
        axes[1, 1].set_title('Classification Accuracy', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Accuracy', fontsize=12)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_loss_components(self, save_path):
        """Plot loss components evolution with subplot labels"""
        if not self.history['loss_components']:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Loss Components Evolution - {self.dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        
        # Add subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        positions = [(0,0), (0,1), (1,0), (1,1)]
        
        for label, pos in zip(subplot_labels, positions):
            axes[pos].text(-0.1, 1.05, label, transform=axes[pos].transAxes, 
                          fontsize=14, fontweight='bold', va='bottom', ha='right')
        
        loss_components = defaultdict(list)
        for epoch_components in self.history['loss_components']:
            for component, value in epoch_components.items():
                loss_components[component].append(value)
        
        components = list(loss_components.keys())
        colors = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6', '#E67E22']
        
        for i, component in enumerate(components[:4]):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            ax.plot(loss_components[component], label=component, 
                   color=colors[i % len(colors)], linewidth=2)
            ax.set_title(f'{component.capitalize()} Loss', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss Value', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_roc_pr_curves(self, save_path):
        """Plot ROC and Precision-Recall curves with subplot labels"""
        test_metrics = self.evaluate_comprehensive(self.val_loader)
        
        if 'anomaly_scores' not in test_metrics:
            return
        
        y_true = test_metrics['anomaly_true_labels']
        y_scores = test_metrics['anomaly_scores']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'ROC and Precision-Recall Curves - {self.dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        
        # Add subplot labels
        axes[0].text(-0.1, 1.05, '(a)', transform=axes[0].transAxes, 
                    fontsize=14, fontweight='bold', va='bottom', ha='right')
        axes[1].text(-0.1, 1.05, '(b)', transform=axes[1].transAxes, 
                    fontsize=14, fontweight='bold', va='bottom', ha='right')
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = test_metrics['anomaly_auc']
        
        axes[0].plot(fpr, tpr, color='#E74C3C', lw=3, label=f'ROC curve (AUC = {auc_score:.3f})')
        axes[0].plot([0, 1], [0, 1], color='#34495E', lw=2, linestyle='--', label='Random')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0].set_ylabel('True Positive Rate', fontsize=12)
        axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0].legend(loc="lower right", fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap_score = test_metrics['anomaly_ap']
        
        axes[1].plot(recall, precision, color='#27AE60', lw=3, label=f'PR curve (AP = {ap_score:.3f})')
        axes[1].axhline(y=np.mean(y_true), color='#34495E', linestyle='--', label='Random')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Recall', fontsize=12)
        axes[1].set_ylabel('Precision', fontsize=12)
        axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1].legend(loc="lower left", fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_confusion_matrix(self, save_path):
        """Plot confusion matrix"""
        test_metrics = self.evaluate_comprehensive(self.val_loader)
        
        if 'anomaly_predictions' not in test_metrics:
            return
        
        y_true = test_metrics['anomaly_true_labels']
        y_pred = test_metrics['anomaly_predictions']
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'shrink': 0.8},
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title(f'Confusion Matrix - {self.dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        
        # Add percentage annotations
        total = np.sum(cm)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm[i,j]/total*100:.1f}%)', 
                        ha='center', va='center', fontsize=11, color='red', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_error_analysis(self, save_path):
        """Plot detailed error analysis with subplot labels"""
        test_metrics = self.evaluate_comprehensive(self.val_loader)
        
        if 'anomaly_scores' not in test_metrics:
            return
        
        y_true = test_metrics['anomaly_true_labels']
        y_scores = test_metrics['anomaly_scores']
        y_pred = test_metrics['anomaly_predictions']
        
        # Error analysis
        tp_mask = (y_true == 1) & (y_pred == 1)
        tn_mask = (y_true == 0) & (y_pred == 0)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Error Analysis - {self.dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        
        # Add subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        positions = [(0,0), (0,1), (1,0), (1,1)]
        
        for label, pos in zip(subplot_labels, positions):
            axes[pos].text(-0.1, 1.05, label, transform=axes[pos].transAxes, 
                          fontsize=14, fontweight='bold', va='bottom', ha='right')
        
        # Score distributions by prediction type
        axes[0, 0].hist(y_scores[tp_mask], bins=20, alpha=0.7, label='True Positive', color='#27AE60')
        axes[0, 0].hist(y_scores[fn_mask], bins=20, alpha=0.7, label='False Negative', color='#E74C3C')
        axes[0, 0].set_title('Anomaly Score Distribution\n(True vs False Negatives)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Anomaly Score', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(y_scores[tn_mask], bins=20, alpha=0.7, label='True Negative', color='#3498DB')
        axes[0, 1].hist(y_scores[fp_mask], bins=20, alpha=0.7, label='False Positive', color='#F39C12')
        axes[0, 1].set_title('Anomaly Score Distribution\n(True vs False Positives)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Anomaly Score', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error counts
        error_counts = [np.sum(tp_mask), np.sum(tn_mask), np.sum(fp_mask), np.sum(fn_mask)]
        error_labels = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
        colors = ['#27AE60', '#3498DB', '#F39C12', '#E74C3C']
        
        axes[1, 0].bar(error_labels, error_counts, color=colors, alpha=0.8)
        axes[1, 0].set_title('Prediction Counts', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Count', fontsize=12)
        plt.setp(axes[1, 0].get_xticklabels(), rotation=45, fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Threshold analysis
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        
        for threshold in thresholds:
            pred_thresh = (y_scores > threshold).astype(int)
            if np.sum(pred_thresh) > 0:
                from sklearn.metrics import precision_recall_fscore_support
                _, _, f1, _ = precision_recall_fscore_support(y_true, pred_thresh, average='binary', zero_division=0)
            else:
                f1 = 0
            f1_scores.append(f1)
        
        axes[1, 1].plot(thresholds, f1_scores, label='F1 Score', color='#8E44AD', linewidth=3)
        axes[1, 1].set_title('F1 Score vs Threshold', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Threshold', fontsize=12)
        axes[1, 1].set_ylabel('F1 Score', fontsize=12)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _save_metrics_csv(self, csv_dir):
        """Save all metrics to CSV files"""
        csv_dir = Path(csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        training_data = []
        for epoch, (loss, val_metrics, loss_comps) in enumerate(zip(
            self.history['train_loss'], 
            self.history['val_metrics'],
            self.history['loss_components']
        )):
            row = {
                'epoch': epoch + 1,
                'train_loss': loss,
                'val_auc': val_metrics.get('anomaly_auc', 0),
                'val_f1': val_metrics.get('anomaly_f1', 0),
                'val_optimal_f1': val_metrics.get('anomaly_optimal_f1', 0),
                'val_precision': val_metrics.get('anomaly_precision', 0),
                'val_recall': val_metrics.get('anomaly_recall', 0),
                'classification_accuracy': val_metrics.get('classification_accuracy', 0)
            }
            # Add loss components
            row.update(loss_comps)
            training_data.append(row)
        
        pd.DataFrame(training_data).to_csv(csv_dir / f'{self.dataset_name}_training_history.csv', index=False)
        
        # Final test metrics
        final_metrics = self.evaluate_comprehensive(self.val_loader)
        
        if 'anomaly_scores' in final_metrics:
            # Save predictions
            predictions_data = {
                'true_labels': final_metrics['anomaly_true_labels'],
                'predicted_labels': final_metrics['anomaly_predictions'],
                'anomaly_scores': final_metrics['anomaly_scores']
            }
            pd.DataFrame(predictions_data).to_csv(csv_dir / f'{self.dataset_name}_predictions.csv', index=False)
            
            # Save summary metrics
            summary_metrics = {
                'dataset': [self.dataset_name],
                'auc': [final_metrics['anomaly_auc']],
                'f1': [final_metrics['anomaly_f1']],
                'optimal_f1': [final_metrics['anomaly_optimal_f1']],
                'precision': [final_metrics['anomaly_precision']],
                'recall': [final_metrics['anomaly_recall']],
                'accuracy': [final_metrics['classification_accuracy']],
                'best_f1': [self.best_f1],
                'best_auc': [self.best_auc]
            }
            pd.DataFrame(summary_metrics).to_csv(csv_dir / f'{self.dataset_name}_summary.csv', index=False)
        
        logger.info(f"Metrics CSV files saved to: {csv_dir}")
    
    def generate_report(self, test_loader):
        """Generate comprehensive evaluation report"""
        logger.info("Generating comprehensive evaluation report...")
        
        metrics = self.evaluate_comprehensive(test_loader, use_adaptive_threshold=True)
        
        report = {
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'use_physics': getattr(self.model, 'use_physics', False)
            },
            'performance_metrics': {
                'anomaly_detection': {
                    'auc_roc': metrics.get('anomaly_auc', 0),
                    'average_precision': metrics.get('anomaly_ap', 0),
                    'f1_score': metrics.get('anomaly_f1', 0),
                    'optimal_f1': metrics.get('anomaly_optimal_f1', 0),
                    'precision': metrics.get('anomaly_precision', 0),
                    'recall': metrics.get('anomaly_recall', 0),
                    'optimal_threshold': metrics.get('anomaly_optimal_threshold', 0.5)
                },
                'classification': {
                    'accuracy': metrics.get('classification_accuracy', 0)
                }
            },
            'training_history': self.history,
            'best_scores': {
                'best_f1': self.best_f1,
                'best_auc': self.best_auc
            }
        }
        
        return report


def run_enhanced_experiment(dataset_name, device='cuda' if torch.cuda.is_available() else 'cpu',
                          use_physics=True, anomaly_ratio=0.05, num_epochs=200, lr=0.001, output_dir=None,
                          hidden_dim=128, output_dim=64, num_gcn_layers=3, num_attention_heads=4, dropout=0.2):
    """Run enhanced experiment with comprehensive evaluation and automatic plotting"""
    
    logger.info(f"="*60)
    logger.info(f"Starting Enhanced GANTGNN Experiment: {dataset_name}")
    logger.info(f"Physics-aware: {use_physics}, Anomaly ratio: {anomaly_ratio}")
    logger.info(f"Model config: hidden_dim={hidden_dim}, output_dim={output_dim}, layers={num_gcn_layers}")
    logger.info(f"="*60)
    
    loader = DatasetLoader()
    
    if dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
        features, adj_matrix, labels = loader.load_citation_dataset(dataset_name)
    elif dataset_name.lower() == 'reddit':
        features, adj_matrix, labels = loader.load_reddit_dataset(max_nodes=3000)
    elif dataset_name.lower() in ['ogbn-arxiv', 'ogbn_arxiv', 'arxiv']:
        features, adj_matrix, labels = loader.load_ogbn_arxiv_dataset(max_nodes=5000)
    elif dataset_name.lower() in ['books', 'weibo']:
        features, adj_matrix, labels = loader.create_synthetic_dataset(dataset_name, anomaly_ratio)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    logger.info(f"Dataset loaded: {features.shape[0]} nodes, {features.shape[1]} features")
    logger.info(f"Adjacency matrix shape: {adj_matrix.shape}")
    logger.info(f"Number of classes: {len(np.unique(labels))}")
    
    features, adj_matrix, anomaly_labels = loader.inject_enhanced_anomalies(
        features, adj_matrix, anomaly_ratio=anomaly_ratio
    )
    logger.info(f"Anomalies injected: {torch.sum(anomaly_labels).item()} ({torch.mean(anomaly_labels).item()*100:.1f}%)")
    
    dataset = GraphDataset(features, adj_matrix, labels, anomaly_labels, normalize=True)
    
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    input_dim = features.shape[1]
    num_classes = len(np.unique(labels))
    
    lambda_weights = {
        'classification': 1.0,
        'anomaly': 2.5,
        'reconstruction': 0.3,
        'structure': 0.4,
        'physics': 0.7 if use_physics else 0.0,
        'contrastive': 0.4
    }
    
    model = GANTGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_gcn_layers=num_gcn_layers,
        num_classes=num_classes,
        dropout=dropout,
        num_attention_heads=num_attention_heads,
        use_physics=use_physics,
        lambda_weights=lambda_weights
    )
    
    trainer = EnhancedTrainer(model, device, output_dir)
    
    logger.info("Starting training...")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Physics-aware: {use_physics}")
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=1e-4,
        patience=30,
        min_delta=0.001,
        dataset_name=dataset_name
    )
    
    logger.info("Generating final evaluation report...")
    report = trainer.generate_report(test_loader)
    
    anomaly_perf = report['performance_metrics']['anomaly_detection']
    class_perf = report['performance_metrics']['classification']
    
    logger.info(f"\n" + "="*60)
    logger.info(f"FINAL RESULTS - {dataset_name.upper()}")
    logger.info(f"="*60)
    logger.info(f"Anomaly Detection:")
    logger.info(f"  AUC-ROC: {anomaly_perf['auc_roc']:.4f}")
    logger.info(f"  Average Precision: {anomaly_perf['average_precision']:.4f}")
    logger.info(f"  F1 Score: {anomaly_perf['f1_score']:.4f}")
    logger.info(f"  Optimal F1: {anomaly_perf['optimal_f1']:.4f}")
    logger.info(f"  Precision: {anomaly_perf['precision']:.4f}")
    logger.info(f"  Recall: {anomaly_perf['recall']:.4f}")
    logger.info(f"  Optimal Threshold: {anomaly_perf['optimal_threshold']:.3f}")
    logger.info(f"Classification:")
    logger.info(f"  Accuracy: {class_perf['accuracy']:.4f}")
    logger.info(f"Model:")
    logger.info(f"  Parameters: {report['model_info']['total_parameters']:,}")
    logger.info(f"  Physics-aware: {report['model_info']['use_physics']}")
    logger.info(f"="*60)
    
    return {
        'dataset': dataset_name,
        'report': report,
        'model': trainer.model,
        'trainer': trainer,
        'anomaly_ratio': anomaly_ratio,
        'use_physics': use_physics
    }


def run_all_enhanced_experiments(datasets=None, use_physics=True, anomaly_ratio=0.05, output_dir=None,
                                hidden_dim=128, output_dim=64, num_gcn_layers=3, num_attention_heads=4, dropout=0.2):
    """Run enhanced experiments on all datasets with comparison"""
    
    if datasets is None:
        datasets = ['cora', 'citeseer', 'pubmed', 'reddit', 'ogbn-arxiv', 'books', 'weibo']
    
    results = {}
    summary_data = []
    
    logger.info(f"Running experiments on {len(datasets)} datasets")
    logger.info(f"Datasets: {', '.join(datasets)}")
    logger.info(f"Model config: hidden_dim={hidden_dim}, output_dim={output_dim}, layers={num_gcn_layers}")
    
    for dataset in datasets:
        logger.info(f"\n{'='*80}")
        logger.info(f"DATASET: {dataset.upper()}")
        logger.info(f"{'='*80}")
        
        result = run_enhanced_experiment(
            dataset_name=dataset,
            use_physics=use_physics,
            anomaly_ratio=anomaly_ratio,
            num_epochs=150,
            lr=0.001,
            output_dir=output_dir,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_gcn_layers=num_gcn_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout
        )
        
        results[dataset] = result
        
        perf = result['report']['performance_metrics']
        summary_data.append({
            'Dataset': dataset.capitalize(),
            'AUC': perf['anomaly_detection']['auc_roc'],
            'F1': perf['anomaly_detection']['f1_score'],
            'Optimal F1': perf['anomaly_detection']['optimal_f1'],
            'Precision': perf['anomaly_detection']['precision'],
            'Recall': perf['anomaly_detection']['recall'],
            'Accuracy': perf['classification']['accuracy'],
            'Physics': use_physics
        })
        
        filename = f'enhanced_result_{dataset}_physics_{use_physics}.pt'
        torch.save(result, filename)
        logger.info(f"Saved results to {filename}")
    
    all_results_filename = f'enhanced_all_results_physics_{use_physics}.pt'
    torch.save(results, all_results_filename)
    
    print(f"\n{'='*100}")
    print(f"ENHANCED GANTGNN EXPERIMENT SUMMARY")
    print(f"Physics-aware: {use_physics}, Anomaly ratio: {anomaly_ratio*100:.1f}%")
    print(f"{'='*100}")
    print(f"{'Dataset':<12} {'AUC':<8} {'F1':<8} {'Opt F1':<8} {'Precision':<10} {'Recall':<8} {'Accuracy':<10}")
    print(f"{'-'*100}")
    
    total_auc, total_f1, total_acc = 0, 0, 0
    count = 0
    
    for data in summary_data:
        print(f"{data['Dataset']:<12} {data['AUC']:<8.3f} {data['F1']:<8.3f} "
              f"{data['Optimal F1']:<8.3f} {data['Precision']:<10.3f} "
              f"{data['Recall']:<8.3f} {data['Accuracy']:<10.3f}")
        
        total_auc += data['AUC']
        total_f1 += data['Optimal F1']
        total_acc += data['Accuracy']
        count += 1
    
    if count > 0:
        print(f"{'-'*100}")
        print(f"{'Average':<12} {total_auc/count:<8.3f} {'-':<8} "
              f"{total_f1/count:<8.3f} {'-':<10} {'-':<8} {total_acc/count:<10.3f}")
    
    print(f"{'='*100}")
    print(f"Results saved to: {all_results_filename}")
    
    return results, summary_data


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    print("Enhanced GANTGNN Training System")
    print("===============================")
    print("Supported Datasets:")
    print("  • Citation Networks: Cora, CiteSeer, PubMed")
    print("  • Social Networks: Reddit")
    print("  • Academic Networks: OGBN-Arxiv")
    print("  • Synthetic Networks: Books, Weibo")
    print("===============================")
    print("")
    #Cora, CiteSeer, PubMed, Reddit, OGBN-Arxiv, Books, Weibo
    # Option 1: Run single experiment with custom model parameters
    result = run_enhanced_experiment(
        dataset_name='weibo', 
        use_physics=True, 
        anomaly_ratio=0.05, 
        output_dir='./Results',
        hidden_dim=128,
        num_gcn_layers=3,
        num_attention_heads=8
    )
    
    # Option 2: Run all experiments with default parameters
    # results, summary = run_all_enhanced_experiments(
    #     datasets=['cora', 'citeseer', 'pubmed', 'reddit', 'ogbn-arxiv', 'books', 'weibo'],
    #     use_physics=True,
    #     anomaly_ratio=0.05,
    #     output_dir='./Results'
    # )
    
    # Option 3: Comparison with and without physics
    # results_with_physics, _ = run_all_enhanced_experiments(use_physics=True, output_dir='./Results')
    # results_without_physics, _ = run_all_enhanced_experiments(use_physics=False, output_dir='./Results')