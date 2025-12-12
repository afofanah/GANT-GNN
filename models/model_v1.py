import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import math
from typing import Dict, Tuple, Optional, List

class GraphConvolution(nn.Module):
    """GCN layer with normalization and residual connections"""
    
    def __init__(self, in_features, out_features, bias=True, dropout=0.1, use_residual=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_residual = use_residual and (in_features == out_features)
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        if self.use_residual and in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features, bias=False)
        else:
            self.residual_proj = None
            
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        if input.dim() > 2:
            input = input.squeeze(0)
        if adj.dim() > 2:
            adj = adj.squeeze(0)
        
        adj_normalized = self._normalize_adjacency(adj)
        support = torch.mm(input, self.weight)
        output = torch.mm(adj_normalized, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(input)
            else:
                residual = input
            output = output + residual
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output
    
    def _normalize_adjacency(self, adj):
        adj_with_self_loops = adj + torch.eye(adj.size(0), device=adj.device)
        
        degree = torch.sum(adj_with_self_loops, dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.

        degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_normalized = torch.mm(torch.mm(degree_matrix_inv_sqrt, adj_with_self_loops), 
                                 degree_matrix_inv_sqrt)
        
        return adj_normalized


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer with improved attention mechanism"""
    
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.alpha = alpha
        
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2 * self.head_dim, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layer_norm = nn.LayerNorm(out_features)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    
    def forward(self, input, adj):
        N = input.size(0)
        h = torch.mm(input, self.W)  
        h = h.view(N, self.num_heads, self.head_dim)  
        attention_weights = self._compute_attention(h, adj)  
        attention_weights_t = attention_weights.transpose(0, 1)  
        h_t = h.transpose(0, 1)  
        h_prime_t = torch.bmm(attention_weights_t, h_t)  
        h_prime = h_prime_t.transpose(0, 1)  
        h_prime = h_prime.contiguous().view(N, -1)  
        h_prime = self.layer_norm(h_prime)
        
        return h_prime
    
    def _compute_attention(self, h, adj):
        """Compute attention coefficients"""
        N, num_heads, head_dim = h.size()
        if adj.dim() == 3:
            adj = adj.squeeze(0)  
        h_i = h.unsqueeze(2).expand(N, num_heads, N, head_dim)  
        h_j = h.unsqueeze(0).expand(N, N, num_heads, head_dim).permute(1, 2, 0, 3)  

        attention_input = torch.cat([h_i, h_j], dim=-1)  
        e = self.leakyrelu(torch.matmul(attention_input, self.a).squeeze(-1))  
        attention_mask = adj.unsqueeze(1).expand(N, num_heads, N)  
        e = torch.where(attention_mask > 0, e, torch.full_like(e, -9e15))

        attention_weights = F.softmax(e, dim=-1)  
        attention_weights = self.dropout(attention_weights)
        
        return attention_weights

class AnomalyScoreLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64):
        super(AnomalyScoreLayer, self).__init__()
        
        self.feature_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )
        
        self.reconstruction_score = nn.Linear(hidden_dim // 2, 1)
        self.isolation_score = nn.Linear(hidden_dim // 2, 1)
        self.structural_score = nn.Linear(hidden_dim // 2, 1)
        self.score_attention = nn.Linear(3, 3)
        
        self.calibration = nn.Sequential(
            nn.Linear(1, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_embeddings, reconstructed_embeddings, adjacency=None):
            original_shape = node_embeddings.shape
    
            if node_embeddings.dim() == 4:
                node_embeddings = node_embeddings.squeeze(1)
                if reconstructed_embeddings is not None:
                    reconstructed_embeddings = reconstructed_embeddings.squeeze(1)
                if adjacency is not None:
                    adjacency = adjacency.squeeze(1)
            
            features = self.feature_net(node_embeddings)
            
            if reconstructed_embeddings is not None:
                recon_error = F.mse_loss(
                    node_embeddings, reconstructed_embeddings, reduction='none'
                ).mean(dim=-1, keepdim=True)
                batch_max = recon_error.max()
                batch_min = recon_error.min()
                if batch_max > batch_min:
                    recon_error = (recon_error - batch_min) / (batch_max - batch_min)
            else:
                recon_error = torch.zeros(node_embeddings.shape[:-1] + (1,), device=node_embeddings.device)
            
            isolation_score = self.calculate_isolation_score(node_embeddings)

            if adjacency is not None:
                structure_score = self.calculate_structure_score(node_embeddings, adjacency)
            else:
                structure_score = torch.zeros_like(recon_error)
            
            score1 = self.reconstruction_score(features) * recon_error
            score2 = self.isolation_score(features) * isolation_score
            score3 = self.structural_score(features) * structure_score

            score_inputs = torch.cat([score1, score2, score3], dim=-1)
            attention_weights = F.softmax(self.score_attention(score_inputs), dim=-1)
            
            combined_score = (
                attention_weights[..., 0:1] * score1 + 
                attention_weights[..., 1:2] * score2 + 
                attention_weights[..., 2:3] * score3
            )
            if combined_score.dim() > 2:
                combined_score_2d = combined_score.view(-1, 1)
                anomaly_score_2d = self.calibration(combined_score_2d)
                anomaly_score = anomaly_score_2d.view(combined_score.shape)
            else:
                anomaly_score = self.calibration(combined_score)

            if len(original_shape) == 4:
                anomaly_score = anomaly_score.unsqueeze(1)
            
            return anomaly_score
    
    def calculate_isolation_score(self, embeddings):
        normalized = F.normalize(embeddings, p=2, dim=-1)
        if normalized.dim() == 4:
            normalized_3d = normalized.squeeze(1)  
            similarity = torch.bmm(normalized_3d, normalized_3d.transpose(1, 2))
            mask = torch.eye(similarity.size(1), device=similarity.device)
            mask = mask.unsqueeze(0).expand_as(similarity)
            
        elif normalized.dim() == 3:
            similarity = torch.bmm(normalized, normalized.transpose(1, 2))
            mask = torch.eye(similarity.size(1), device=similarity.device)
            mask = mask.unsqueeze(0).expand_as(similarity)
            
        elif normalized.dim() == 2:
            similarity = torch.mm(normalized, normalized.transpose(0, 1))
            mask = torch.eye(similarity.size(0), device=similarity.device)
            
        else:
            raise ValueError(f"Unsupported tensor dimension: {normalized.dim()}")
        
        similarity = similarity * (1 - mask)
        avg_sim = similarity.sum(dim=-1, keepdim=True) / (similarity.size(-1) - 1)
        isolation_score = 1 - avg_sim
        if embeddings.dim() == 4 and isolation_score.dim() == 2:
            isolation_score = isolation_score.unsqueeze(1)  
            
        return isolation_score
    
    def calculate_structure_score(self, embeddings, adjacency):
        if adjacency.dim() == 4:
            adj_3d = adjacency.squeeze(1)
            degree = adj_3d.sum(dim=-1, keepdim=True)
            neighbor_degree = torch.bmm(adj_3d, degree)
            
        elif adjacency.dim() == 3:
            degree = adjacency.sum(dim=-1, keepdim=True)
            neighbor_degree = torch.bmm(adjacency, degree)
            
        elif adjacency.dim() == 2:
            degree = adjacency.sum(dim=-1, keepdim=True)
            neighbor_degree = torch.mm(adjacency, degree)
            
        else:
            raise ValueError(f"Unsupported adjacency tensor dimension: {adjacency.dim()}")
        
        avg_neighbor_degree = neighbor_degree / (degree + 1e-8)
        degree_ratio = degree / (avg_neighbor_degree + 1e-8)
        normalized_ratio = torch.sigmoid((degree_ratio - 1) * 3)
    
        if embeddings.dim() == 4 and normalized_ratio.dim() == 2:
            normalized_ratio = normalized_ratio.unsqueeze(1)  
            
        return normalized_ratio

class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.3):
        super(GraphAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed
    
class TAD(nn.Module):
    """physics/thermodynamic components analyzer"""
    
    def __init__(self, config):
        super(TAD, self).__init__()
        self.config = config
        self.temperature = config.get('temperature', 1.0)
        self.consistency_history = []
        
    def compute_physics_components(self, node_embeddings, adj_matrix, anomaly_scores=None):
        """Compute physics components"""
        device = node_embeddings.device

        if node_embeddings.dim() == 3:
            node_embeddings = node_embeddings.squeeze(0)
        if adj_matrix.dim() == 3:
            adj_matrix = adj_matrix.squeeze(0)
        if anomaly_scores is not None and anomaly_scores.dim() == 2:
            anomaly_scores = anomaly_scores.squeeze(0)
        elif anomaly_scores is None:
            anomaly_scores = torch.zeros(node_embeddings.size(0), device=device)
        
        components = {}
        components['total_energy'] = self._compute_total_energy(node_embeddings, adj_matrix)
        components['local_entropy'] = self._compute_local_entropy(node_embeddings, adj_matrix)
        components['equilibrium_deviation'] = self._compute_equilibrium_deviation(node_embeddings, anomaly_scores)

        components['structural_centrality'] = self._compute_structural_centrality(node_embeddings, adj_matrix)
        components['information_flow'] = self._compute_information_flow(node_embeddings, adj_matrix)
        components['stability_measure'] = self._compute_stability_measure(node_embeddings, adj_matrix)
        components['graph_curvature'] = self._compute_graph_curvature(node_embeddings, adj_matrix)
        
        return components
    
    def _compute_total_energy(self, node_embeddings, adj_matrix):
        """total energy computation"""
        kinetic = torch.sum(node_embeddings ** 2, dim=-1)
        neighbor_embeddings = torch.mm(adj_matrix, node_embeddings)
        potential = -torch.sum(node_embeddings * neighbor_embeddings, dim=-1)
        degrees = torch.sum(adj_matrix, dim=-1)
        interaction_energy = potential / (degrees + 1e-8)
        
        total_energy = kinetic + interaction_energy
        return total_energy
    
    def _compute_local_entropy(self, node_embeddings, adj_matrix):
        """local entropy with neighbourhood consideration"""
        normalized_embeddings = F.normalize(node_embeddings, p=2, dim=-1)

        local_entropies = []
        for i in range(node_embeddings.size(0)):
            neighbors = torch.nonzero(adj_matrix[i]).squeeze(-1)
            if len(neighbors) > 0:
                neighbor_embeddings = normalized_embeddings[neighbors]
                similarities = torch.mm(neighbor_embeddings, normalized_embeddings[i].unsqueeze(-1)).squeeze()

                probs = F.softmax(similarities / self.temperature, dim=0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                local_entropies.append(entropy)
            else:
                local_entropies.append(torch.tensor(0.0, device=node_embeddings.device))
        
        return torch.stack(local_entropies)
    
    def _compute_equilibrium_deviation(self, node_embeddings, anomaly_scores):
        """Enhanced equilibrium deviation computation"""
        global_mean = torch.mean(node_embeddings, dim=0, keepdim=True)

        local_means = []
        for i in range(node_embeddings.size(0)):
            distances = torch.norm(node_embeddings - node_embeddings[i], dim=1)
            k = min(5, node_embeddings.size(0) - 1)
            _, top_k_indices = torch.topk(distances, k + 1, largest=False)
            neighbors = top_k_indices[1:]  
            
            local_mean = torch.mean(node_embeddings[neighbors], dim=0, keepdim=True)
            local_means.append(local_mean)
        
        local_means = torch.cat(local_means, dim=0)
        global_deviation = torch.norm(node_embeddings - global_mean, p=2, dim=-1)
        local_deviation = torch.norm(node_embeddings - local_means, p=2, dim=-1)
        combined_deviation = 0.6 * global_deviation + 0.4 * local_deviation
        
        return combined_deviation
    
    def _compute_structural_centrality(self, node_embeddings, adj_matrix):
        """Compute structural centrality based on embeddings and topology"""
        degrees = torch.sum(adj_matrix, dim=-1)
        degree_centrality = degrees / (adj_matrix.size(0) - 1)
        embedding_norms = torch.norm(node_embeddings, p=2, dim=-1)
        embedding_centrality = embedding_norms / torch.max(embedding_norms)
        centrality = 0.5 * degree_centrality + 0.5 * embedding_centrality
        
        return centrality
    
    def _compute_information_flow(self, node_embeddings, adj_matrix):
        """Compute information flow through the graph"""
        similarities = torch.mm(F.normalize(node_embeddings, p=2, dim=-1),
                               F.normalize(node_embeddings, p=2, dim=-1).t())
        
        flow_capacities = []
        for i in range(node_embeddings.size(0)):
            neighbors = torch.nonzero(adj_matrix[i]).squeeze(-1)
            if len(neighbors) > 0:
                neighbor_similarities = similarities[i, neighbors]
                flow_capacity = torch.var(neighbor_similarities)
                flow_capacities.append(flow_capacity)
            else:
                flow_capacities.append(torch.tensor(0.0, device=node_embeddings.device))
        
        return torch.stack(flow_capacities)
    
    def _compute_stability_measure(self, node_embeddings, adj_matrix):
        """Compute node stability measure"""
        stabilities = []
        for i in range(node_embeddings.size(0)):
            neighbors = torch.nonzero(adj_matrix[i]).squeeze(-1)
            if len(neighbors) > 1:
                neighbor_embeddings = node_embeddings[neighbors]
                stability = 1.0 / (1.0 + torch.var(neighbor_embeddings, dim=0).mean())
                stabilities.append(stability)
            else:
                stabilities.append(torch.tensor(1.0, device=node_embeddings.device))
        
        return torch.stack(stabilities)
    
    def _compute_graph_curvature(self, node_embeddings, adj_matrix):
        """Compute discrete graph curvature"""
        curvatures = []
        
        for i in range(node_embeddings.size(0)):
            neighbors = torch.nonzero(adj_matrix[i]).squeeze(-1)
            if len(neighbors) >= 2:
                node_embed = node_embeddings[i]
                neighbor_embeds = node_embeddings[neighbors]
                centered_embeds = neighbor_embeds - node_embed.unsqueeze(0)

                if len(neighbors) >= 2:
                    angles = []
                    for j in range(len(neighbors)):
                        for k in range(j + 1, len(neighbors)):
                            cos_angle = F.cosine_similarity(
                                centered_embeds[j].unsqueeze(0),
                                centered_embeds[k].unsqueeze(0)
                            )
                            angles.append(cos_angle)
                    
                    if angles:
                        mean_angle = torch.stack(angles).mean()
                        curvature = 1.0 - mean_angle  
                        curvatures.append(curvature)
                    else:
                        curvatures.append(torch.tensor(0.0, device=node_embeddings.device))
                else:
                    curvatures.append(torch.tensor(0.0, device=node_embeddings.device))
            else:
                curvatures.append(torch.tensor(0.0, device=node_embeddings.device))
        
        return torch.stack(curvatures)


class AdaptiveThresholdSelector:
    """Adaptive threshold selection for optimal F1 score"""
    
    def __init__(self, method='f1_optimal'):
        self.method = method
        self.threshold_history = []
        
    def select_threshold(self, scores, labels=None, validation_split=0.2):
        """Select optimal threshold based on validation data"""
        if labels is None:
            return self._statistical_threshold(scores)

        if len(scores) <= 5:
            return self._f1_optimal_threshold(scores, labels)

        n_val = max(1, int(len(scores) * validation_split))
        n_val = min(n_val, len(scores) - 1)  
        
        if n_val >= len(scores):
            return self._f1_optimal_threshold(scores, labels)
        
        val_indices = np.random.choice(len(scores), n_val, replace=False)
        val_scores = scores[val_indices]
        val_labels = labels[val_indices]
        
        if self.method == 'f1_optimal':
            return self._f1_optimal_threshold(val_scores, val_labels)
        elif self.method == 'precision_recall':
            return self._precision_recall_threshold(val_scores, val_labels)
        else:
            return self._youden_index_threshold(val_scores, val_labels)
    
    def _f1_optimal_threshold(self, scores, labels):
        """Find threshold that maximizes F1 score"""
        thresholds = np.linspace(0, 1, 100)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            predictions = (scores >= threshold).astype(int)
            if len(np.unique(predictions)) > 1:  
                f1 = f1_score(labels, predictions)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        return best_threshold
    
    def _precision_recall_threshold(self, scores, labels):
        """Find threshold based on precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    def _youden_index_threshold(self, scores, labels):
        """Find threshold using Youden's index"""
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        return thresholds[best_idx]
    
    def _statistical_threshold(self, scores):
        """Statistical threshold when no labels available"""
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        threshold = median + 2.5 * mad  
        return min(threshold, 0.9) 


class GANTGNN(nn.Module):
    """GANTGNN with improved architecture and loss functions"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers, num_classes,
                 dropout=0.2, num_attention_heads=4, use_physics=True, lambda_weights=None):
        super(GANTGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_gcn_layers = num_gcn_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_attention_heads = num_attention_heads
        self.use_physics = use_physics

        default_weights = {
            'classification': 1.0,
            'anomaly': 1.0,
            'reconstruction': 0.5,
            'structure': 0.3,
            'physics': 0.4,
            'contrastive': 0.2
        }
        self.lambda_weights = lambda_weights or default_weights
        self._build_enhanced_model()
        
        # Physics components
        if self.use_physics:
            physics_config = {
                'temperature': 1.0,
                'alpha': 0.1,
                'beta': 0.2
            }
            self.physics_analyzer = TAD(physics_config)
        self.threshold_selector = AdaptiveThresholdSelector(method='f1_optimal')
        
    def _build_enhanced_model(self):
        self.gcn_layers = nn.ModuleList()

        self.gcn_layers.append(
            GraphConvolution(self.input_dim, self.hidden_dim, dropout=self.dropout)
        )
        
        # Hidden layers
        for _ in range(self.num_gcn_layers - 2):
            self.gcn_layers.append(
                GraphConvolution(self.hidden_dim, self.hidden_dim, dropout=self.dropout)
            )
        
        # Output layer
        if self.num_gcn_layers > 1:
            self.gcn_layers.append(
                GraphConvolution(self.hidden_dim, self.output_dim, dropout=self.dropout)
            )
        
        final_dim = self.output_dim if self.num_gcn_layers > 1 else self.hidden_dim
 
        self.graph_attention = GraphAttentionLayer(
            final_dim, final_dim, 
            num_heads=self.num_attention_heads, 
            dropout=self.dropout
        )
        
        # autoencoder with skip connections
        self.encoder = nn.Sequential(
            nn.Linear(final_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.output_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, final_dim)
        )
        
        # Multi-task heads
        self.classifier = self._build_mlp_head(final_dim, self.num_classes, 'classification')
        self.structure_detector = self._build_mlp_head(final_dim, 1, 'anomaly')
        self.feature_detector = self._build_mlp_head(final_dim, 1, 'anomaly')
        
        if self.use_physics:
            self.physics_detector = self._build_mlp_head(final_dim, 1, 'anomaly')
            fusion_input = 3
        else:
            fusion_input = 2
        
        # fusion network
        self.anomaly_fusion = nn.Sequential(
            nn.Linear(fusion_input, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ELU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Contrastive learning projection head
        self.projection_head = nn.Sequential(
            nn.Linear(final_dim, self.hidden_dim // 2),
            nn.ELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4)
        )
        
    def _build_mlp_head(self, input_dim, output_dim, task_type):
        """Build task-specific MLP head"""
        if task_type == 'classification':
            return nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim // 2),
                nn.LayerNorm(self.hidden_dim // 2),
                nn.ELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, output_dim)
            )
        else:  # anomaly detection
            return nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim // 2),
                nn.LayerNorm(self.hidden_dim // 2),
                nn.ELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, output_dim),
                nn.Sigmoid()
            )
    
    def forward(self, features, adj_matrix, labels=None, anomaly_labels=None):
        batch_size = features.size(0)
        if batch_size == 1:
            features = features.squeeze(0)
            adj_matrix = adj_matrix.squeeze(0)
        
        x = features
        gcn_outputs = []
        
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, adj_matrix)
            if i < len(self.gcn_layers) - 1:
                x = F.elu(x)
            gcn_outputs.append(x)

        attended_x = self.graph_attention(x, adj_matrix)
        encoded = self.encoder(attended_x)
        reconstructed = self.decoder(encoded)

        logits = self.classifier(attended_x)
        structure_scores = self.structure_detector(attended_x).squeeze(-1)
        feature_scores = self.feature_detector(attended_x).squeeze(-1)
        
        # Physics-based detection
        physics_components = {}
        physics_scores = None
        
        if self.use_physics:
            physics_components = self.physics_analyzer.compute_physics_components(
                attended_x, adj_matrix, structure_scores
            )
            physics_scores = self.physics_detector(attended_x).squeeze(-1)
        
        # Anomaly score fusion
        if self.use_physics and physics_scores is not None:
            fusion_input = torch.stack([structure_scores, feature_scores, physics_scores], dim=-1)
        else:
            fusion_input = torch.stack([structure_scores, feature_scores], dim=-1)
        
        fused_anomaly_scores = self.anomaly_fusion(fusion_input).squeeze(-1)
        projections = self.projection_head(attended_x)

        outputs = {
            'gcn_embeddings': gcn_outputs[-2] if len(gcn_outputs) > 1 else gcn_outputs[0],
            'final_embeddings': attended_x,
            'reconstructed': reconstructed,
            'logits': logits,
            'structure_scores': structure_scores,
            'feature_scores': feature_scores,
            'physics_scores': physics_scores,
            'fused_anomaly_scores': fused_anomaly_scores,
            'anomaly_scores': fused_anomaly_scores,
            'physics_components': physics_components,
            'projections': projections
        }

        if batch_size == 1:
            for key in outputs:
                if outputs[key] is not None and isinstance(outputs[key], torch.Tensor):
                    if outputs[key].dim() == 1:
                        outputs[key] = outputs[key].unsqueeze(0)
                    elif outputs[key].dim() == 2:
                        outputs[key] = outputs[key].unsqueeze(0)

        if self.training and (labels is not None or anomaly_labels is not None):
            total_loss, loss_components = self._compute_enhanced_loss(
                outputs, labels, anomaly_labels, features, adj_matrix
            )
            outputs['loss'] = total_loss
            outputs['loss_components'] = loss_components
        
        return outputs
    
    def _compute_enhanced_loss(self, outputs, labels, anomaly_labels, features, adj_matrix):
        """Compute multi-objective loss"""
        total_loss = 0
        loss_components = {}

        def flatten_if_needed(tensor):
            if tensor is None:
                return None
            if tensor.dim() == 3:  
                return tensor.squeeze(0)
            elif tensor.dim() == 2 and tensor.size(0) == 1:  
                return tensor.squeeze(0)
            elif tensor.dim() == 1:  
                return tensor
            else:
                return tensor
        
        if labels is not None:
            logits = flatten_if_needed(outputs['logits'])
            labels_flat = flatten_if_needed(labels)
  
            if labels_flat.dim() > 1:
                labels_flat = labels_flat.view(-1)

            if logits.size(0) != labels_flat.size(0):
                min_size = min(logits.size(0), labels_flat.size(0))
                logits = logits[:min_size]
                labels_flat = labels_flat[:min_size]
            
            max_label = torch.max(labels_flat)
            if max_label >= self.num_classes:
                unique_labels = torch.unique(labels_flat)
                if len(unique_labels) <= self.num_classes:
                    label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}
                    labels_flat = torch.tensor([label_mapping[label.item()] for label in labels_flat], 
                                            device=labels_flat.device, dtype=labels_flat.dtype)
                else:
                    labels_flat = torch.clamp(labels_flat, 0, self.num_classes - 1)
                
            cls_loss = F.cross_entropy(logits, labels_flat.long())
            total_loss += self.lambda_weights['classification'] * cls_loss
            loss_components['classification_loss'] = cls_loss.item()
        # if labels is not None:
        #     logits = flatten_if_needed(outputs['logits'])
        #     labels_flat = flatten_if_needed(labels)
            
        #     # Ensure labels are 1D
        #     if labels_flat.dim() > 1:
        #         labels_flat = labels_flat.view(-1)
            
        #     # Make sure dimensions match
        #     if logits.size(0) != labels_flat.size(0):
        #         min_size = min(logits.size(0), labels_flat.size(0))
        #         logits = logits[:min_size]
        #         labels_flat = labels_flat[:min_size]
            
        #     cls_loss = F.cross_entropy(logits, labels_flat.long())
        #     total_loss += self.lambda_weights['classification'] * cls_loss
        #     loss_components['classification_loss'] = cls_loss.item()
        
        # Anomaly detection loss with focal loss
        if anomaly_labels is not None:
            anomaly_scores = flatten_if_needed(outputs['fused_anomaly_scores'])
            anomaly_labels_flat = flatten_if_needed(anomaly_labels).float()

            if anomaly_scores.size(0) != anomaly_labels_flat.size(0):
                min_size = min(anomaly_scores.size(0), anomaly_labels_flat.size(0))
                anomaly_scores = anomaly_scores[:min_size]
                anomaly_labels_flat = anomaly_labels_flat[:min_size]
            
            # Focal loss for imbalanced anomaly detection
            alpha = 0.75
            gamma = 2.0
            ce_loss = F.binary_cross_entropy(anomaly_scores, anomaly_labels_flat, reduction='none')
            pt = torch.where(anomaly_labels_flat == 1, anomaly_scores, 1 - anomaly_scores)
            focal_loss = alpha * (1 - pt) ** gamma * ce_loss
            anomaly_loss = focal_loss.mean()
            
            total_loss += self.lambda_weights['anomaly'] * anomaly_loss
            loss_components['anomaly_loss'] = anomaly_loss.item()
            
        # Reconstruction loss with dimension matching
        gcn_embeddings = flatten_if_needed(outputs['gcn_embeddings'])
        reconstructed = flatten_if_needed(outputs['reconstructed'])

        min_dim = min(gcn_embeddings.size(-1), reconstructed.size(-1))
        gcn_embeddings_matched = gcn_embeddings[:, :min_dim]
        reconstructed_matched = reconstructed[:, :min_dim]

        recon_l2 = F.mse_loss(reconstructed_matched, gcn_embeddings_matched)
        cos_sim = F.cosine_similarity(reconstructed_matched, gcn_embeddings_matched, dim=-1)
        perceptual_loss = (1 - cos_sim).mean()

        recon_loss = recon_l2 + 0.3 * perceptual_loss
        total_loss += self.lambda_weights['reconstruction'] * recon_loss
        loss_components['reconstruction_loss'] = recon_loss.item()
        
        # Contrastive learning loss
        if anomaly_labels is not None:
            projections = flatten_if_needed(outputs['projections'])
            contrastive_loss = self._compute_contrastive_loss(projections, anomaly_labels)
            total_loss += self.lambda_weights['contrastive'] * contrastive_loss
            loss_components['contrastive_loss'] = contrastive_loss.item()
        
        # Physics-aware loss
        if self.use_physics and anomaly_labels is not None:
            physics_loss = self._compute_physics_loss(outputs['physics_components'], anomaly_labels)
            total_loss += self.lambda_weights['physics'] * physics_loss
            loss_components['physics_loss'] = physics_loss.item()
        
        return total_loss, loss_components
    
    def _compute_enhanced_structure_loss(self, embeddings, adj_matrix):
        """Enhanced structure preservation with multiple scales"""
        # Local structure preservation
        embedding_sim = torch.mm(F.normalize(embeddings, p=2, dim=-1),
                                F.normalize(embeddings, p=2, dim=-1).t())
        local_structure_loss = F.mse_loss(torch.sigmoid(embedding_sim), adj_matrix.float())
        
        # Global structure preservation (using random walk)
        degree_matrix = torch.diag(torch.sum(adj_matrix, dim=-1))
        laplacian = degree_matrix - adj_matrix
        
        # Normalized Laplacian
        degree_inv_sqrt = torch.diag(torch.pow(torch.sum(adj_matrix, dim=-1) + 1e-8, -0.5))
        normalized_laplacian = torch.mm(torch.mm(degree_inv_sqrt, laplacian), degree_inv_sqrt)
        
        # Spectral loss
        embedding_laplacian = torch.mm(torch.mm(embeddings.t(), normalized_laplacian), embeddings)
        spectral_loss = torch.trace(embedding_laplacian) / embeddings.size(0)
        
        return local_structure_loss + 0.1 * spectral_loss
    
    def _compute_contrastive_loss(self, projections, anomaly_labels, temperature=0.07):
        """Compute supervised contrastive loss"""
        projections = F.normalize(projections, p=2, dim=-1)
        similarity_matrix = torch.mm(projections, projections.t()) / temperature

        anomaly_labels_flat = anomaly_labels.squeeze(0) if anomaly_labels.dim() == 2 else anomaly_labels
        labels_eq = anomaly_labels_flat.unsqueeze(0) == anomaly_labels_flat.unsqueeze(1)

        mask = torch.eye(labels_eq.size(0), device=labels_eq.device).bool()
        labels_eq = labels_eq & ~mask

        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=-1, keepdim=True))

        mean_log_prob_pos = (labels_eq * log_prob).sum(dim=-1) / (labels_eq.sum(dim=-1) + 1e-8)
        
        return -mean_log_prob_pos.mean()
    
    def _compute_physics_loss(self, physics_components, anomaly_labels):
        """Enhanced physics-aware loss"""
        anomaly_labels_flat = anomaly_labels.squeeze(0) if anomaly_labels.dim() == 2 else anomaly_labels
        anomaly_mask = anomaly_labels_flat.bool()
        normal_mask = ~anomaly_mask
        
        total_physics_loss = 0
        
        # Energy-based separation
        if 'total_energy' in physics_components:
            energy = physics_components['total_energy']
            if anomaly_mask.sum() > 0 and normal_mask.sum() > 0:
                energy_sep = torch.mean(energy[normal_mask]) - torch.mean(energy[anomaly_mask])
                total_physics_loss += F.relu(energy_sep + 0.1)
        
        # Entropy-based separation
        if 'local_entropy' in physics_components:
            entropy = physics_components['local_entropy']
            if anomaly_mask.sum() > 0 and normal_mask.sum() > 0:
                entropy_sep = torch.mean(entropy[normal_mask]) - torch.mean(entropy[anomaly_mask])
                total_physics_loss += F.relu(entropy_sep + 0.1)
        
        # Stability-based loss
        if 'stability_measure' in physics_components:
            stability = physics_components['stability_measure']
            if anomaly_mask.sum() > 0:
                # Anomalies should have lower stability
                stability_loss = torch.mean(stability[anomaly_mask])
                total_physics_loss += stability_loss
        
        return total_physics_loss
    
    def detect_anomalies(self, features, adj_matrix, use_adaptive_threshold=True, 
                        validation_labels=None):
        """Enhanced anomaly detection with adaptive thresholding"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(features, adj_matrix)
            
            fused_scores = outputs['fused_anomaly_scores'].squeeze()
            physics_scores = outputs.get('physics_scores')
            if physics_scores is not None:
                physics_scores = physics_scores.squeeze()
            
            # Adaptive threshold selection
            if use_adaptive_threshold and validation_labels is not None:
                scores_np = fused_scores.cpu().numpy()
                labels_np = validation_labels.cpu().numpy().flatten()

                if len(scores_np) > 1 and len(labels_np) > 1 and len(scores_np) == len(labels_np):
                    threshold = self.threshold_selector.select_threshold(scores_np, labels_np)
                else:
                    threshold = 0.5  
            else:
                threshold = 0.5
            
            predictions = (fused_scores > threshold).float()
            ranked_indices = torch.argsort(fused_scores, descending=True)
         
            explanations = {
                'threshold_used': threshold,
                'confidence_scores': fused_scores,
                'ranking': ranked_indices,
                'physics_explanations': outputs.get('physics_components', {}),
                'structure_contributions': outputs['structure_scores'].squeeze(),
                'feature_contributions': outputs['feature_scores'].squeeze()
            }
            
            if physics_scores is not None:
                explanations['physics_contributions'] = physics_scores
            
            return {
                'fused_scores': fused_scores,
                'predictions': predictions,
                'explanations': explanations,
                'confidence': torch.std(fused_scores) if fused_scores.numel() > 1 else torch.tensor(0.0)
            }


def create_enhanced_model(input_dim, hidden_dim=128, output_dim=64, num_gcn_layers=3, 
                         num_classes=2, dropout=0.2, num_attention_heads=4, 
                         use_physics=True, lambda_weights=None):
    """Factory function to create enhanced model with optimal hyperparameters"""

    if lambda_weights is None:
        lambda_weights = {
            'classification': 1.0,
            'anomaly': 2.0,        
            'reconstruction': 0.3,  
            'structure': 0.4,       
            'physics': 0.6,        
            'contrastive': 0.3     
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
    
    return model


class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class LearningRateScheduler:
    """Custom learning rate scheduler"""
    def __init__(self, optimizer, mode='cosine', patience=10, factor=0.5, min_lr=1e-6):
        self.optimizer = optimizer
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.counter = 0
        self.best_score = None
        
    def step(self, score):
        if self.mode == 'plateau':
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self._reduce_lr()
                    self.counter = 0
    
    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            print(f"Reducing learning rate: {old_lr:.6f} -> {new_lr:.6f}")


def train_enhanced_model(model, train_loader, val_loader, num_epochs=200, lr=0.001):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = LearningRateScheduler(optimizer, mode='plateau')
    early_stopping = EarlyStopping(patience=30, min_delta=0.001)
    
    best_f1 = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            features, adj_matrix, labels, anomaly_labels = batch
            outputs = model(features, adj_matrix, labels, anomaly_labels)
            
            loss = outputs['loss']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_scores = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                features, adj_matrix, _, anomaly_labels = batch
                results = model.detect_anomalies(features, adj_matrix, 
                                               validation_labels=anomaly_labels)
                
                val_scores.extend(results['fused_scores'].cpu().numpy())
                val_labels.extend(anomaly_labels.squeeze().cpu().numpy())
        
        val_scores = np.array(val_scores)
        val_labels = np.array(val_labels)
        
        auc = roc_auc_score(val_labels, val_scores)
        threshold = 0.5  
        predictions = (val_scores > threshold).astype(int)
        f1 = f1_score(val_labels, predictions)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
        scheduler.step(f1)

        if early_stopping(f1, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if f1 > best_f1:
            best_f1 = f1
    
    return model, best_f1