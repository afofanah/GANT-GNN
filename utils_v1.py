import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
import networkx as nx
from collections import defaultdict
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib parameters for consistent, clean plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0
})

class ResultAnalyzer:
    """Comprehensive result analysis and visualization with statistical reporting"""
    
    def __init__(self, results_dict=None, output_dir='./Results'):
        self.results = results_dict or {}
        self.comparison_data = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        self.colors = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6', '#E67E22', '#1ABC9C']
        
    def load_results(self, filepath):
        """Load results from file"""
        self.results = torch.load(filepath, map_location='cpu')
        return self.results
    
    def load_statistical_results(self, filepath):
        """Load results with statistical data (mean±std format)"""
        with open(filepath, 'r') as f:
            self.statistical_results = json.load(f)
        return self.statistical_results
    
    def plot_statistical_comparison(self, results_data, save_path=None):
        """Plot comparison with error bars for statistical results with subplot labels"""
        datasets = list(results_data.keys())
        metrics = ['auc', 'f1', 'optimal_f1', 'precision', 'recall', 'accuracy']
        metric_names = ['AUC-ROC', 'F1 Score', 'Optimal F1', 'Precision', 'Recall', 'Accuracy']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('GANTGNN: Statistical Performance Analysis', fontsize=16, fontweight='bold')
        
        # Add subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
        
        for label, pos in zip(subplot_labels, positions):
            axes[pos].text(-0.1, 1.05, label, transform=axes[pos].transAxes, 
                          fontsize=14, fontweight='bold', va='bottom', ha='right')
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            means = []
            stds = []
            dataset_names = []
            
            for dataset in datasets:
                if 'stats' in results_data[dataset]:
                    stats_data = results_data[dataset]['stats'][metric]
                    means.append(stats_data['mean'])
                    stds.append(stats_data['std'])
                    dataset_names.append(dataset.capitalize())
            
            if means:
                x = np.arange(len(dataset_names))
                bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, 
                             color=self.colors[i % len(self.colors)])
                
                ax.set_xlabel('Datasets', fontsize=12)
                ax.set_ylabel(name, fontsize=12)
                ax.set_title(f'{name} (Mean ± Std)', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(dataset_names, rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for j, (mean, std) in enumerate(zip(means, stds)):
                    ax.text(j, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_loss_components_evolution(self, results, dataset_name, save_path=None):
        """Plot evolution of different loss components during training with subplot labels"""
        if dataset_name not in results:
            print(f"Dataset {dataset_name} not found in results")
            return
        
        trainer = results[dataset_name].get('trainer')
        if not trainer or not hasattr(trainer, 'history'):
            print("Training history not available")
            return
            
        history = trainer.history

        if 'loss_components' not in history or not history['loss_components']:
            print("Loss components not tracked in training history")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Loss Components Evolution - {dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        
        # Add subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        positions = [(0,0), (0,1), (1,0), (1,1)]
        
        for label, pos in zip(subplot_labels, positions):
            axes[pos].text(-0.1, 1.05, label, transform=axes[pos].transAxes, 
                          fontsize=14, fontweight='bold', va='bottom', ha='right')
        
        loss_components = defaultdict(list)
        for epoch_components in history['loss_components']:
            for component, value in epoch_components.items():
                loss_components[component].append(value)
        
        # Plot each component
        components = list(loss_components.keys())
        for i, component in enumerate(components[:4]):  
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            ax.plot(loss_components[component], label=component, 
                   color=self.colors[i % len(self.colors)], linewidth=2)
            ax.set_title(f'{component.capitalize()} Loss', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss Value', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_physics_analysis(self, results, dataset_name, save_path=None):
        """Analyze physics-aware components contribution with subplot labels"""
        if dataset_name not in results:
            print(f"Dataset {dataset_name} not found in results")
            return
        
        if not results[dataset_name].get('use_physics', False):
            print("Physics components not enabled for this experiment")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Physics-Aware Components Analysis - {dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        
        # Add subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        positions = [(0,0), (0,1), (1,0), (1,1)]
        
        for label, pos in zip(subplot_labels, positions):
            axes[pos].text(-0.1, 1.05, label, transform=axes[pos].transAxes, 
                          fontsize=14, fontweight='bold', va='bottom', ha='right')
        
        # Physics loss evolution (if tracked)
        trainer = results[dataset_name].get('trainer')
        if trainer and hasattr(trainer, 'history'):
            history = trainer.history
            if 'physics_metrics' in history:
                physics_metrics = history['physics_metrics']
                
                # Energy conservation
                if 'energy_conservation' in physics_metrics:
                    axes[0, 0].plot(physics_metrics['energy_conservation'], 
                                   color=self.colors[0], linewidth=2)
                    axes[0, 0].set_title('Energy Conservation Over Time', fontsize=14, fontweight='bold')
                    axes[0, 0].set_xlabel('Epoch', fontsize=12)
                    axes[0, 0].set_ylabel('Energy Deviation', fontsize=12)
                    axes[0, 0].grid(True, alpha=0.3)
                
                # Entropy analysis
                if 'entropy_analysis' in physics_metrics:
                    axes[0, 1].plot(physics_metrics['entropy_analysis'], 
                                   color=self.colors[1], linewidth=2)
                    axes[0, 1].set_title('System Entropy Evolution', fontsize=14, fontweight='bold')
                    axes[0, 1].set_xlabel('Epoch', fontsize=12)
                    axes[0, 1].set_ylabel('Entropy', fontsize=12)
                    axes[0, 1].grid(True, alpha=0.3)
                
                # Equilibrium deviation
                if 'equilibrium_deviation' in physics_metrics:
                    axes[1, 0].plot(physics_metrics['equilibrium_deviation'], 
                                   color=self.colors[2], linewidth=2)
                    axes[1, 0].set_title('Equilibrium Deviation', fontsize=14, fontweight='bold')
                    axes[1, 0].set_xlabel('Epoch', fontsize=12)
                    axes[1, 0].set_ylabel('Deviation', fontsize=12)
                    axes[1, 0].grid(True, alpha=0.3)
                
                # Physics weight adaptation
                if 'adaptive_physics_weight' in physics_metrics:
                    axes[1, 1].plot(physics_metrics['adaptive_physics_weight'], 
                                   color=self.colors[3], linewidth=2)
                    axes[1, 1].set_title('Adaptive Physics Weight', fontsize=14, fontweight='bold')
                    axes[1, 1].set_xlabel('Epoch', fontsize=12)
                    axes[1, 1].set_ylabel('Weight', fontsize=12)
                    axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_embedding_analysis(self, results, dataset_name, save_path=None):
        """Comprehensive embedding quality analysis with subplot labels"""
        if dataset_name not in results:
            print(f"Dataset {dataset_name} not found in results")
            return

        trainer = results[dataset_name].get('trainer')
        model = results[dataset_name].get('model')
        
        if not trainer or not model:
            print("Trainer or model not available for embedding analysis")
            return

        sample_batch = next(iter(trainer.val_loader))
        features = sample_batch['features'].to(trainer.device)
        adj_matrix = sample_batch['adj_matrix'].to(trainer.device)
        labels = sample_batch['labels']
        anomaly_labels = sample_batch.get('anomaly_labels')
        
        if anomaly_labels is None:
            print("No anomaly labels available for embedding analysis")
            return
        
        # Convert to numpy and ensure proper shapes and types
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(anomaly_labels, torch.Tensor):
            anomaly_labels = anomaly_labels.cpu().numpy()
            
        # Ensure 1D arrays
        labels = np.asarray(labels).flatten().astype(int)
        anomaly_labels = np.asarray(anomaly_labels).flatten().astype(int)
        
        with torch.no_grad():
            model.eval()
            outputs = model(features, adj_matrix)
            embeddings = outputs.get('final_embeddings', outputs.get('gcn_embeddings', features))
            embeddings = embeddings.cpu().numpy().squeeze()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Embedding Analysis - {dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        
        # Add subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
        
        for label, pos in zip(subplot_labels, positions):
            axes[pos].text(-0.1, 1.05, label, transform=axes[pos].transAxes, 
                          fontsize=14, fontweight='bold', va='bottom', ha='right')
        
        if embeddings.shape[0] > 50 and embeddings.shape[1] > 1:  # Only if we have enough points and features
            perplexity = min(30, max(5, embeddings.shape[0] // 4))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(embeddings)
            scatter = axes[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                        c=labels, cmap='tab10', alpha=0.7, s=20)
            axes[0, 0].set_title('t-SNE by Node Classes', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('t-SNE 1', fontsize=14)
            axes[0, 0].set_ylabel('t-SNE 2', fontsize=14)
            plt.colorbar(scatter, ax=axes[0, 0])
            
            # Color by anomaly labels using numpy operations
            color_array = np.where(anomaly_labels == 0, '#3498DB', '#E74C3C')
            axes[0, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=color_array, alpha=0.7, s=20)
            axes[0, 1].set_title('t-SNE by Anomaly Status', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('t-SNE 1', fontsize=14)
            axes[0, 1].set_ylabel('t-SNE 2', fontsize=14)
            
            # Create legend for anomaly plot
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#3498DB', label='Normal'),
                             Patch(facecolor='#E74C3C', label='Anomaly')]
            axes[0, 1].legend(handles=legend_elements, fontsize=14)
        else:
            axes[0, 0].text(0.5, 0.5, 'Not enough data\nfor t-SNE analysis', 
                           ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)
            axes[0, 1].text(0.5, 0.5, 'Not enough data\nfor t-SNE analysis', 
                           ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
        
        # PCA analysis
        if embeddings.shape[1] > 1:  # Need at least 2 features for PCA
            pca = PCA()
            embeddings_pca = pca.fit_transform(embeddings)
            
            # Explained variance
            axes[0, 2].plot(np.cumsum(pca.explained_variance_ratio_), 'bo-', linewidth=2, markersize=4)
            axes[0, 2].set_title('PCA Explained Variance', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Component', fontsize=14)
            axes[0, 2].set_ylabel('Cumulative Explained Variance', fontsize=14)
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'Not enough features\nfor PCA analysis', 
                           ha='center', va='center', transform=axes[0, 2].transAxes, fontsize=14)
        
        # Embedding magnitude distribution
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        normal_indices = np.where(anomaly_labels == 0)[0]
        anomaly_indices = np.where(anomaly_labels == 1)[0]
        
        normal_norms = embedding_norms[normal_indices]
        anomaly_norms = embedding_norms[anomaly_indices]
        
        if len(normal_norms) > 0:
            axes[1, 0].hist(normal_norms, bins=20, alpha=0.7, label='Normal', color='#3498DB')
        if len(anomaly_norms) > 0:
            axes[1, 0].hist(anomaly_norms, bins=20, alpha=0.7, label='Anomaly', color='#E74C3C')
        axes[1, 0].set_title('Embedding Magnitude Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('L2 Norm', fontsize=14)
        axes[1, 0].set_ylabel('Frequency', fontsize=14)
        axes[1, 0].legend(fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Clustering analysis
        if embeddings.shape[0] > 10:
            n_clusters = min(5, len(np.unique(labels)))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Cluster purity with respect to anomaly labels
                cluster_purity = []
                for cluster_id in range(n_clusters):
                    cluster_indices = np.where(cluster_labels == cluster_id)[0]
                    cluster_size = len(cluster_indices)
                    if cluster_size > 0:
                        cluster_anomaly_labels = anomaly_labels[cluster_indices]
                        cluster_anomaly_ratio = np.mean(cluster_anomaly_labels)
                        cluster_purity.append(cluster_anomaly_ratio)
                    else:
                        cluster_purity.append(0)
                
                if len(cluster_purity) > 0:
                    axes[1, 1].bar(range(len(cluster_purity)), cluster_purity, 
                                  color=self.colors[:len(cluster_purity)], alpha=0.8)
                    axes[1, 1].set_title('Cluster Anomaly Ratio', fontsize=14, fontweight='bold')
                    axes[1, 1].set_xlabel('Cluster ID', fontsize=14)
                    axes[1, 1].set_ylabel('Anomaly Ratio', fontsize=14)
                    axes[1, 1].grid(True, alpha=0.3)
                else:
                    axes[1, 1].text(0.5, 0.5, 'Clustering failed', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=14)
            else:
                axes[1, 1].text(0.5, 0.5, 'Not enough clusters\nfor analysis', 
                               ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=14)
        else:
            axes[1, 1].text(0.5, 0.5, 'Not enough data\nfor clustering analysis', 
                           ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=14)
        
        # Embedding similarity analysis
        if embeddings.shape[0] > 1:
            similarities = np.dot(embeddings, embeddings.T)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            similarities = similarities / (norms @ norms.T + 1e-8)
            
            normal_indices = np.where(anomaly_labels == 0)[0]
            anomaly_indices = np.where(anomaly_labels == 1)[0]
            
            normal_count = len(normal_indices)
            anomaly_count = len(anomaly_indices)
            
            if normal_count > 1 and anomaly_count > 1:
                normal_sim = similarities[np.ix_(normal_indices, normal_indices)]
                anomaly_sim = similarities[np.ix_(anomaly_indices, anomaly_indices)]
                cross_sim = similarities[np.ix_(normal_indices, anomaly_indices)]
                
                # Extract upper triangular parts to avoid duplicates
                normal_sim_vals = normal_sim[np.triu_indices_from(normal_sim, k=1)]
                anomaly_sim_vals = anomaly_sim[np.triu_indices_from(anomaly_sim, k=1)]
                cross_sim_vals = cross_sim.flatten()
                
                sim_data = [normal_sim_vals, anomaly_sim_vals, cross_sim_vals]
                
                # Only plot if we have data
                if all(len(data) > 0 for data in sim_data):
                    box_plot = axes[1, 2].boxplot(sim_data, labels=['Normal-Normal', 'Anomaly-Anomaly', 'Normal-Anomaly'])
                    axes[1, 2].set_title('Embedding Similarity Analysis', fontsize=14, fontweight='bold')
                    axes[1, 2].set_ylabel('Cosine Similarity', fontsize=14)
                    axes[1, 2].grid(True, alpha=0.3)
                    plt.setp(axes[1, 2].get_xticklabels(), rotation=45, fontsize=14)
                else:
                    axes[1, 2].text(0.5, 0.5, 'Insufficient data\nfor similarity analysis', 
                                   ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=14)
            else:
                axes[1, 2].text(0.5, 0.5, 'Insufficient data\nfor similarity analysis', 
                               ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=14)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_hyperparameter_sensitivity(self, hyperopt_results, save_path=None):
        """Analyze hyperparameter sensitivity from optimization results with subplot labels"""
        if 'all_trials' not in hyperopt_results:
            print("No hyperparameter optimization data found")
            return
        
        trials = hyperopt_results['all_trials']
        if not trials:
            return

        params = list(trials[0]['config'].keys())
        scores = [trial['score'] for trial in trials]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # Add subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
        
        for label, pos in zip(subplot_labels, positions):
            axes[pos].text(-0.1, 1.05, label, transform=axes[pos].transAxes, 
                          fontsize=14, fontweight='bold', va='bottom', ha='right')
        
        for i, param in enumerate(params[:6]):  
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            param_values = [trial['config'][param] for trial in trials]
            
            if isinstance(param_values[0], (int, float)):
                # Numerical parameter - scatter plot
                ax.scatter(param_values, scores, alpha=0.7, color=self.colors[i % len(self.colors)], s=30)
                ax.set_xlabel(param.replace('_', ' ').title(), fontsize=12)
                ax.set_ylabel('Performance Score', fontsize=12)
                ax.set_title(f'{param.replace("_", " ").title()} Sensitivity', fontsize=14, fontweight='bold')
                
                # Add trend line
                if len(set(param_values)) > 1:
                    z = np.polyfit(param_values, scores, 1)
                    p = np.poly1d(z)
                    ax.plot(sorted(param_values), p(sorted(param_values)), "r--", alpha=0.8, linewidth=2)
            else:
                # Categorical parameter - box plot
                unique_values = list(set(param_values))
                grouped_scores = [[] for _ in unique_values]
                
                for val, score in zip(param_values, scores):
                    grouped_scores[unique_values.index(val)].append(score)
                
                ax.boxplot(grouped_scores, labels=unique_values)
                ax.set_xlabel(param.replace('_', ' ').title(), fontsize=12)
                ax.set_ylabel('Performance Score', fontsize=12)
                ax.set_title(f'{param.replace("_", " ").title()} Sensitivity', fontsize=14, fontweight='bold')
                plt.setp(ax.get_xticklabels(), rotation=45, fontsize=12)
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_convergence_analysis(self, results, save_path=None):
        """Analyze convergence patterns across datasets with subplot labels"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Convergence Analysis', fontsize=16, fontweight='bold')
        
        # Add subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        positions = [(0,0), (0,1), (1,0), (1,1)]
        
        for label, pos in zip(subplot_labels, positions):
            axes[pos].text(-0.1, 1.05, label, transform=axes[pos].transAxes, 
                          fontsize=14, fontweight='bold', va='bottom', ha='right')
        
        # Collect convergence data
        convergence_data = {}
        for dataset, result in results.items():
            trainer = result.get('trainer')
            if trainer and hasattr(trainer, 'history'):
                history = trainer.history
                convergence_data[dataset] = {
                    'loss': history['train_loss'],
                    'val_auc': [m.get('anomaly_auc', 0) for m in history['val_metrics']],
                    'val_f1': [m.get('anomaly_f1', 0) for m in history['val_metrics']]
                }
        
        # Training loss convergence
        for i, (dataset, data) in enumerate(convergence_data.items()):
            axes[0, 0].plot(data['loss'], label=dataset.capitalize(), alpha=0.8, 
                           color=self.colors[i % len(self.colors)], linewidth=2)
        axes[0, 0].set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation AUC convergence
        for i, (dataset, data) in enumerate(convergence_data.items()):
            axes[0, 1].plot(data['val_auc'], label=dataset.capitalize(), alpha=0.8, 
                           color=self.colors[i % len(self.colors)], linewidth=2)
        axes[0, 1].set_title('Validation AUC Convergence', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('AUC', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Convergence speed analysis
        convergence_epochs = []
        dataset_names = []
        for dataset, data in convergence_data.items():
            # Find epoch where AUC reaches 95% of final value
            final_auc = max(data['val_auc']) if data['val_auc'] else 0
            target_auc = 0.95 * final_auc
            convergence_epoch = next((i for i, auc in enumerate(data['val_auc']) if auc >= target_auc), len(data['val_auc']))
            convergence_epochs.append(convergence_epoch)
            dataset_names.append(dataset.capitalize())
        
        axes[1, 0].bar(dataset_names, convergence_epochs, color=self.colors[:len(dataset_names)], alpha=0.8)
        axes[1, 0].set_title('Convergence Speed (Epochs to 95% Final AUC)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Dataset', fontsize=12)
        axes[1, 0].set_ylabel('Epochs', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        plt.setp(axes[1, 0].get_xticklabels(), rotation=45, fontsize=10)
        
        # Final performance vs convergence speed
        final_aucs = [max(data['val_auc']) if data['val_auc'] else 0 for data in convergence_data.values()]
        scatter = axes[1, 1].scatter(convergence_epochs, final_aucs, s=100, alpha=0.7, c=self.colors[:len(dataset_names)])
        for i, dataset in enumerate(dataset_names):
            axes[1, 1].annotate(dataset, (convergence_epochs[i], final_aucs[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=10)
        axes[1, 1].set_title('Performance vs Convergence Speed', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Convergence Epochs', fontsize=12)
        axes[1, 1].set_ylabel('Final AUC', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_statistical_significance(self, results1, results2, labels=None, save_path=None):
        """Test and visualize statistical significance between two experiments with subplot labels"""
        if labels is None:
            labels = ['Experiment 1', 'Experiment 2']
        
        comparison = self.compare_models(results1, results2)
        metrics = ['auc', 'f1', 'precision', 'recall', 'accuracy']
        metric_names = ['AUC', 'F1', 'Precision', 'Recall', 'Accuracy']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Statistical Significance Analysis: {labels[0]} vs {labels[1]}', fontsize=16, fontweight='bold')
        
        # Add subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
        positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]
        
        for label, pos in zip(subplot_labels, positions):
            axes[pos].text(-0.1, 1.05, label, transform=axes[pos].transAxes, 
                          fontsize=14, fontweight='bold', va='bottom', ha='right')
        
        significance_results = {}
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            row = i // 3
            col = i % 3
            ax = axes[row, col] if i < 5 else axes[1, 2]
            
            values1 = comparison['physics_disabled'][metric]
            values2 = comparison['physics_enabled'][metric]
            
            # Perform statistical tests
            values1 = np.asarray(values1).flatten().astype(float)
            values2 = np.asarray(values2).flatten().astype(float)
            
            # Only perform tests if we have valid data
            if len(values1) > 1 and len(values2) > 1:
                t_stat, t_pvalue = ttest_ind(values1, values2)
                u_stat, u_pvalue = mannwhitneyu(values1, values2, alternative='two-sided')
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(values1) + np.var(values2)) / 2)
                if pooled_std > 0:
                    effect_size = (np.mean(values2) - np.mean(values1)) / pooled_std
                else:
                    effect_size = 0
            else:
                t_stat, t_pvalue = 0, 1.0
                u_stat, u_pvalue = 0, 1.0
                effect_size = 0
            
            significance_results[metric] = {
                't_test': {'statistic': t_stat, 'p_value': t_pvalue},
                'mann_whitney': {'statistic': u_stat, 'p_value': u_pvalue},
                'effect_size': effect_size
            }
            
            # Box plot comparison
            box_plot = ax.boxplot([values1, values2], labels=labels)
            ax.set_title(f'{name}\np-value: {t_pvalue:.4f}', fontsize=14, fontweight='bold')
            ax.set_ylabel(name, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add significance indicator
            if t_pvalue < 0.001:
                sig_text = '***'
            elif t_pvalue < 0.01:
                sig_text = '**'
            elif t_pvalue < 0.05:
                sig_text = '*'
            else:
                sig_text = 'n.s.'
            
            ax.text(0.5, 0.95, sig_text, transform=ax.transAxes, 
                   ha='center', va='top', fontsize=16, fontweight='bold')
        
        # Hide unused subplot
        if len(metrics) < 6:
            axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        return significance_results
    
    def plot_error_analysis(self, results, dataset_name, save_path=None):
        """Detailed error analysis for anomaly detection with subplot labels"""
        if dataset_name not in results:
            print(f"Dataset {dataset_name} not found in results")
            return
        
        trainer = results[dataset_name].get('trainer')
        if not trainer:
            print("Trainer not available for error analysis")
            return
            
        test_metrics = trainer.evaluate_comprehensive(trainer.val_loader)
        
        if 'anomaly_scores' not in test_metrics:
            print("Detailed prediction data not available")
            return
        
        y_true = test_metrics['anomaly_true_labels']
        y_scores = test_metrics['anomaly_scores']
        
        # Ensure proper data types and shapes
        y_true = np.asarray(y_true).flatten().astype(int)
        y_scores = np.asarray(y_scores).flatten().astype(float)
        y_pred = test_metrics['anomaly_predictions']
        
        # Convert to numpy arrays and ensure proper types
        y_true = np.array(y_true, dtype=int)
        y_scores = np.array(y_scores, dtype=float)
        y_pred = np.array(y_pred, dtype=int)
        
        # Error analysis
        tp_mask = (y_true == 1) & (y_pred == 1)
        tn_mask = (y_true == 0) & (y_pred == 0)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Error Analysis - {dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        
        # Add subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
        
        for label, pos in zip(subplot_labels, positions):
            axes[pos].text(-0.1, 1.05, label, transform=axes[pos].transAxes, 
                          fontsize=14, fontweight='bold', va='bottom', ha='right')
        
        # Define all indices for error analysis
        tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]  # True Positives
        tn_indices = np.where((y_true == 0) & (y_pred == 0))[0]  # True Negatives
        fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]  # False Positives
        fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]  # False Negatives
        
        # Score distributions by prediction type
        if len(tp_indices) > 0:
            axes[0, 0].hist(y_scores[tp_indices], bins=20, alpha=0.7, label='True Positive', color='#27AE60')
        if len(fn_indices) > 0:
            axes[0, 0].hist(y_scores[fn_indices], bins=20, alpha=0.7, label='False Negative', color='#E74C3C')
        axes[0, 0].set_title('Anomaly Score Distribution\n(True vs False Negatives)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Anomaly Score', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        if len(tn_indices) > 0:
            axes[0, 1].hist(y_scores[tn_indices], bins=20, alpha=0.7, label='True Negative', color='#3498DB')
        if len(fp_indices) > 0:
            axes[0, 1].hist(y_scores[fp_indices], bins=20, alpha=0.7, label='False Positive', color='#F39C12')
        axes[0, 1].set_title('Anomaly Score Distribution\n(True vs False Positives)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Anomaly Score', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error counts
        error_counts = [len(tp_indices), len(tn_indices), len(fp_indices), len(fn_indices)]
        error_labels = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
        colors = ['#27AE60', '#3498DB', '#F39C12', '#E74C3C']
        
        axes[0, 2].bar(error_labels, error_counts, color=colors, alpha=0.8)
        axes[0, 2].set_title('Prediction Counts', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('Count', fontsize=12)
        plt.setp(axes[0, 2].get_xticklabels(), rotation=45, fontsize=10)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Score thresholds analysis
        thresholds = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            pred_thresh = (y_scores > threshold).astype(int)
            tp = np.sum((y_true == 1) & (pred_thresh == 1))
            fp = np.sum((y_true == 0) & (pred_thresh == 1))
            fn = np.sum((y_true == 1) & (pred_thresh == 0))
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0
                
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0
                
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        axes[1, 0].plot(thresholds, precisions, label='Precision', color='#3498DB', linewidth=2)
        axes[1, 0].plot(thresholds, recalls, label='Recall', color='#E74C3C', linewidth=2)
        axes[1, 0].plot(thresholds, f1_scores, label='F1 Score', color='#27AE60', linewidth=2)
        axes[1, 0].set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Threshold', fontsize=12)
        axes[1, 0].set_ylabel('Score', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # ROC curve with error analysis
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        axes[1, 1].plot(fpr, tpr, color='#E74C3C', lw=3)
        axes[1, 1].plot([0, 1], [0, 1], color='#34495E', lw=2, linestyle='--')
        axes[1, 1].set_xlabel('False Positive Rate', fontsize=12)
        axes[1, 1].set_ylabel('True Positive Rate', fontsize=12)
        axes[1, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Precision-Recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
        axes[1, 2].plot(recall_curve, precision_curve, color='#27AE60', lw=3)
        axes[1, 2].set_xlabel('Recall', fontsize=12)
        axes[1, 2].set_ylabel('Precision', fontsize=12)
        axes[1, 2].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_dataset_characteristics(self, results, save_path=None):
        """Analyze relationship between dataset characteristics and performance with subplot labels"""
        dataset_stats = {}
        performance_stats = {}
        
        for dataset, result in results.items():
            # Extract dataset characteristics
            dataset_stats[dataset] = {
                'num_nodes': result.get('num_nodes', 1000),
                'num_features': result.get('num_features', 64),
                'num_anomalies': result.get('num_anomalies', 50),
                'density': 0.1,  # Would need to compute from adjacency matrix
                'clustering_coeff': 0.3  # Would need to compute from graph
            }
            
            # Extract performance
            perf = result['report']['performance_metrics']['anomaly_detection']
            performance_stats[dataset] = {
                'auc': perf['auc_roc'],
                'f1': perf['optimal_f1']
            }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Dataset Characteristics vs Performance', fontsize=16, fontweight='bold')
        
        # Add subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
        positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]
        
        for label, pos in zip(subplot_labels, positions):
            axes[pos].text(-0.1, 1.05, label, transform=axes[pos].transAxes, 
                          fontsize=14, fontweight='bold', va='bottom', ha='right')
        
        characteristics = ['num_nodes', 'num_features', 'num_anomalies', 'density', 'clustering_coeff']
        char_names = ['Nodes', 'Features', 'Anomalies', 'Density', 'Clustering Coeff']
        
        for i, (char, name) in enumerate(zip(characteristics, char_names)):
            row = i // 3
            col = i % 3
            if i >= 5:
                break
            ax = axes[row, col]
            
            x_values = [dataset_stats[d][char] for d in dataset_stats.keys()]
            y_auc = [performance_stats[d]['auc'] for d in dataset_stats.keys()]
            y_f1 = [performance_stats[d]['f1'] for d in dataset_stats.keys()]
            
            ax.scatter(x_values, y_auc, label='AUC', alpha=0.7, color='#3498DB', s=50)
            ax.scatter(x_values, y_f1, label='F1', alpha=0.7, color='#E74C3C', s=50)
            
            # Add dataset labels
            for j, dataset in enumerate(dataset_stats.keys()):
                ax.annotate(dataset, (x_values[j], y_auc[j]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax.set_xlabel(name, fontsize=12)
            ax.set_ylabel('Performance Score', fontsize=12)
            ax.set_title(f'Performance vs {name}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplot
        axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, results_with_physics, results_without_physics=None, 
                                    hyperopt_results=None, output_dir=None):
        """Generate comprehensive analysis report with all visualizations"""
        if output_dir is None:
            output_dir = self.output_dir / 'comprehensive_reports'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating comprehensive analysis report in {output_dir}")
        
        # Basic performance plots for each dataset
        for dataset in results_with_physics.keys():
            dataset_dir = output_dir / dataset
            dataset_dir.mkdir(exist_ok=True)
            
            # Training curves
            self.plot_training_curves(results_with_physics, dataset, 
                                    dataset_dir / 'training_curves.pdf')
            
            # Loss components
            self.plot_loss_components_evolution(results_with_physics, dataset,
                                              dataset_dir / 'loss_components.pdf')
            
            # Physics analysis (if applicable)
            if results_with_physics[dataset].get('use_physics', False):
                self.plot_physics_analysis(results_with_physics, dataset,
                                          dataset_dir / 'physics_analysis.pdf')
            
            # Embedding analysis
            self.plot_embedding_analysis(results_with_physics, dataset,
                                        dataset_dir / 'embedding_analysis.pdf')
            
            # Error analysis
            self.plot_error_analysis(results_with_physics, dataset,
                                   dataset_dir / 'error_analysis.pdf')
            
            # ROC/PR curves
            self.plot_roc_pr_curves(results_with_physics, dataset,
                                  dataset_dir / 'roc_pr_curves.pdf')
            
            # Confusion matrix
            self.plot_confusion_matrix(results_with_physics, dataset,
                                     dataset_dir / 'confusion_matrix.pdf')
        
        # Overall analysis plots
        self.plot_convergence_analysis(results_with_physics, 
                                     output_dir / 'convergence_analysis.pdf')
        
        self.plot_dataset_characteristics(results_with_physics,
                                        output_dir / 'dataset_characteristics.pdf')
        
        # Comparison plots (if baseline results available)
        if results_without_physics:
            comparison = self.compare_models(results_with_physics, results_without_physics)
            self.plot_comparison(comparison, output_dir / 'model_comparison.pdf')
            self.plot_improvement_heatmap(comparison, output_dir / 'improvement_heatmap.pdf')
            
            # Statistical significance analysis
            significance = self.plot_statistical_significance(
                results_with_physics, results_without_physics,
                labels=['Physics-Aware', 'Standard'],
                save_path=output_dir / 'statistical_significance.pdf'
            )
        
        # Hyperparameter analysis (if available)
        if hyperopt_results:
            self.plot_hyperparameter_sensitivity(hyperopt_results,
                                                output_dir / 'hyperparameter_sensitivity.pdf')
        
        print(f"Comprehensive report generated successfully!")
        print(f"Check {output_dir} for all visualizations and analyses.")
    
    def compare_models(self, results_with_physics, results_without_physics):
        """Compare physics-aware vs non-physics models"""
        comparison = {
            'datasets': [],
            'physics_enabled': {'auc': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': []},
            'physics_disabled': {'auc': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': []},
            'improvements': {'auc': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': []}
        }
        
        common_datasets = set(results_with_physics.keys()) & set(results_without_physics.keys())
        
        for dataset in common_datasets:
            with_physics = results_with_physics[dataset]['report']['performance_metrics']
            without_physics = results_without_physics[dataset]['report']['performance_metrics']
            
            comparison['datasets'].append(dataset)
            
            # Extract metrics
            anom_with = with_physics['anomaly_detection']
            anom_without = without_physics['anomaly_detection']
            class_with = with_physics['classification']
            class_without = without_physics['classification']
            
            # Physics enabled
            comparison['physics_enabled']['auc'].append(anom_with['auc_roc'])
            comparison['physics_enabled']['f1'].append(anom_with['optimal_f1'])
            comparison['physics_enabled']['precision'].append(anom_with['precision'])
            comparison['physics_enabled']['recall'].append(anom_with['recall'])
            comparison['physics_enabled']['accuracy'].append(class_with['accuracy'])
            
            # Physics disabled
            comparison['physics_disabled']['auc'].append(anom_without['auc_roc'])
            comparison['physics_disabled']['f1'].append(anom_without['optimal_f1'])
            comparison['physics_disabled']['precision'].append(anom_without['precision'])
            comparison['physics_disabled']['recall'].append(anom_without['recall'])
            comparison['physics_disabled']['accuracy'].append(class_without['accuracy'])
            
            # Improvements
            comparison['improvements']['auc'].append(
                anom_with['auc_roc'] - anom_without['auc_roc']
            )
            comparison['improvements']['f1'].append(
                anom_with['optimal_f1'] - anom_without['optimal_f1']
            )
            comparison['improvements']['precision'].append(
                anom_with['precision'] - anom_without['precision']
            )
            comparison['improvements']['recall'].append(
                anom_with['recall'] - anom_without['recall']
            )
            comparison['improvements']['accuracy'].append(
                class_with['accuracy'] - class_without['accuracy']
            )
        
        return comparison
    
    def plot_comparison(self, comparison_data, save_path=None):
        """Plot model comparison results with subplot labels"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('GANTGNN: Physics-Aware vs Standard Model Comparison', fontsize=16, fontweight='bold')
        
        # Add subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
        positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]
        
        for label, pos in zip(subplot_labels, positions):
            axes[pos].text(-0.1, 1.05, label, transform=axes[pos].transAxes, 
                          fontsize=14, fontweight='bold', va='bottom', ha='right')
        
        metrics = ['auc', 'f1', 'precision', 'recall', 'accuracy']
        metric_names = ['AUC-ROC', 'F1 Score', 'Precision', 'Recall', 'Classification Accuracy']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            datasets = comparison_data['datasets']
            with_physics = comparison_data['physics_enabled'][metric]
            without_physics = comparison_data['physics_disabled'][metric]
            
            x = np.arange(len(datasets))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, without_physics, width, label='Standard', alpha=0.8, color='#E74C3C')
            bars2 = ax.bar(x + width/2, with_physics, width, label='Physics-Aware', alpha=0.8, color='#3498DB')
            
            ax.set_xlabel('Datasets', fontsize=12)
            ax.set_ylabel(name, fontsize=12)
            ax.set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([d.capitalize() for d in datasets], rotation=45, fontsize=10)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            add_value_labels(bars1)
            add_value_labels(bars2)
        
        # Hide the last subplot if we have an odd number of metrics
        if len(metrics) % 3 != 0:
            axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_improvement_heatmap(self, comparison_data, save_path=None):
        """Plot improvement heatmap"""
        datasets = comparison_data['datasets']
        metrics = ['auc', 'f1', 'precision', 'recall', 'accuracy']
        metric_names = ['AUC', 'F1', 'Precision', 'Recall', 'Accuracy']
        
        # Create improvement matrix
        improvement_matrix = []
        for dataset in datasets:
            row = []
            for metric in metrics:
                idx = comparison_data['datasets'].index(dataset)
                improvement = comparison_data['improvements'][metric][idx]
                row.append(improvement)
            improvement_matrix.append(row)
        
        improvement_matrix = np.array(improvement_matrix)
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(improvement_matrix, 
                   xticklabels=metric_names,
                   yticklabels=[d.capitalize() for d in datasets],
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   center=0,
                   cbar_kws={'label': 'Performance Improvement'})
        
        plt.title('Physics-Aware Enhancement: Performance Improvements by Dataset and Metric', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Datasets', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        return improvement_matrix
    
    def plot_training_curves(self, results, dataset_name, save_path=None):
        """Plot training curves for a specific dataset with subplot labels"""
        if dataset_name not in results:
            print(f"Dataset {dataset_name} not found in results")
            return
        
        trainer = results[dataset_name].get('trainer')
        if not trainer or not hasattr(trainer, 'history'):
            print("Training history not available")
            return
            
        history = trainer.history
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Training Curves - {dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        
        # Add subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        positions = [(0,0), (0,1), (1,0), (1,1)]
        
        for label, pos in zip(subplot_labels, positions):
            axes[pos].text(-0.1, 1.05, label, transform=axes[pos].transAxes, 
                          fontsize=14, fontweight='bold', va='bottom', ha='right')
        
        # Training loss
        axes[0, 0].plot(history['train_loss'], color='#2E86C1', linewidth=2)
        axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation metrics over time
        epochs = range(len(history['val_metrics']))
        val_aucs = [m.get('anomaly_auc', 0) for m in history['val_metrics']]
        val_f1s = [m.get('anomaly_f1', 0) for m in history['val_metrics']]
        
        axes[0, 1].plot(epochs, val_aucs, label='AUC', marker='o', markersize=4, color='#E74C3C', linewidth=2)
        axes[0, 1].plot(epochs, val_f1s, label='F1', marker='s', markersize=4, color='#27AE60', linewidth=2)
        axes[0, 1].set_title('Validation Metrics', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Score', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate evolution
        axes[1, 0].text(0.5, 0.5, 'Learning Rate\nHistory Not Available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Learning Rate Evolution', fontsize=14, fontweight='bold')
        
        # Loss components summary
        axes[1, 1].text(0.5, 0.5, 'Loss Components\nAvailable in Training Log', 
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Loss Components Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_pr_curves(self, results, dataset_name, save_path=None):
        """Plot ROC and Precision-Recall curves with subplot labels"""
        if dataset_name not in results:
            print(f"Dataset {dataset_name} not found in results")
            return
        
        trainer = results[dataset_name].get('trainer')
        if not trainer:
            print("Trainer not available")
            return
            
        test_metrics = trainer.evaluate_comprehensive(trainer.val_loader)
        
        if 'anomaly_scores' not in test_metrics:
            print("Anomaly scores not available for plotting curves")
            return
        
        y_true = test_metrics['anomaly_true_labels']
        y_scores = test_metrics['anomaly_scores']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'ROC and Precision-Recall Curves - {dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        
        # Add subplot labels
        axes[0].text(-0.1, 1.05, '(a)', transform=axes[0].transAxes, 
                    fontsize=14, fontweight='bold', va='bottom', ha='right')
        axes[1].text(-0.1, 1.05, '(b)', transform=axes[1].transAxes, 
                    fontsize=14, fontweight='bold', va='bottom', ha='right')
        
        # ROC Curve
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
        
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, results, dataset_name, save_path=None):
        """Plot confusion matrix for anomaly detection"""
        if dataset_name not in results:
            print(f"Dataset {dataset_name} not found in results")
            return
        
        trainer = results[dataset_name].get('trainer')
        if not trainer:
            print("Trainer not available")
            return
            
        test_metrics = trainer.evaluate_comprehensive(trainer.val_loader)
        
        if 'anomaly_predictions' not in test_metrics:
            print("Predictions not available for confusion matrix")
            return
        
        y_true = test_metrics['anomaly_true_labels']
        y_pred = test_metrics['anomaly_predictions']
        
        # Ensure proper data types and shapes
        y_true = np.asarray(y_true).flatten().astype(int)
        y_pred = np.asarray(y_pred).flatten().astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'shrink': 0.8},
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title(f'Confusion Matrix - {dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        
        # Add percentage annotations
        total = np.sum(cm)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm[i,j]/total*100:.1f}%)', 
                        ha='center', va='center', fontsize=11, color='red', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()


def quick_performance_summary(results_file, print_details=True):
    """Quick performance summary from results file with statistical format"""
    if isinstance(results_file, str):
        if results_file.endswith('.json'):
            # JSON file with statistical data
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            summary = []
            for dataset, result in results.items():
                if 'stats' in result:
                    stats = result['stats']
                    summary.append({
                        'Dataset': dataset.capitalize(),
                        'AUC': f"{stats['auc']['mean']:.4f}±{stats['auc']['std']:.3f}",
                        'F1': f"{stats['f1']['mean']:.4f}±{stats['f1']['std']:.3f}",
                        'Optimal F1': f"{stats['optimal_f1']['mean']:.4f}±{stats['optimal_f1']['std']:.3f}",
                        'Precision': f"{stats['precision']['mean']:.4f}±{stats['precision']['std']:.3f}",
                        'Recall': f"{stats['recall']['mean']:.4f}±{stats['recall']['std']:.3f}",
                        'Accuracy': f"{stats['accuracy']['mean']:.4f}±{stats['accuracy']['std']:.3f}"
                    })
        else:
            # PyTorch file with single run data
            results = torch.load(results_file, map_location='cpu')
            
            summary = []
            for dataset, result in results.items():
                perf = result['report']['performance_metrics']
                summary.append({
                    'Dataset': dataset.capitalize(),
                    'AUC': f"{perf['anomaly_detection']['auc_roc']:.4f}",
                    'F1': f"{perf['anomaly_detection']['f1_score']:.4f}",
                    'Optimal F1': f"{perf['anomaly_detection']['optimal_f1']:.4f}",
                    'Precision': f"{perf['anomaly_detection']['precision']:.4f}",
                    'Recall': f"{perf['anomaly_detection']['recall']:.4f}",
                    'Accuracy': f"{perf['classification']['accuracy']:.4f}"
                })
    
    if print_details:
        df = pd.DataFrame(summary)
        print("Performance Summary:")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
    
    return summary


def compare_experiment_results(results_file1, results_file2, labels=None):
    """Compare two experiment results with statistical significance"""
    if labels is None:
        labels = ['Experiment 1', 'Experiment 2']
    
    if results_file1.endswith('.json'):
        with open(results_file1, 'r') as f:
            results1 = json.load(f)
        with open(results_file2, 'r') as f:
            results2 = json.load(f)
    else:
        results1 = torch.load(results_file1, map_location='cpu')
        results2 = torch.load(results_file2, map_location='cpu')
    
    analyzer = ResultAnalyzer()
    comparison = analyzer.compare_models(results1, results2)
    
    print(f"Comparison: {labels[0]} vs {labels[1]}")
    print("=" * 80)
    
    df_comparison = pd.DataFrame({
        'Dataset': [d.capitalize() for d in comparison['datasets']],
        f'{labels[0]} AUC': [f"{x:.4f}" for x in comparison['physics_disabled']['auc']],
        f'{labels[1]} AUC': [f"{x:.4f}" for x in comparison['physics_enabled']['auc']],
        'AUC Δ': [f"{x:+.4f}" for x in comparison['improvements']['auc']],
        f'{labels[0]} F1': [f"{x:.4f}" for x in comparison['physics_disabled']['f1']],
        f'{labels[1]} F1': [f"{x:.4f}" for x in comparison['physics_enabled']['f1']],
        'F1 Δ': [f"{x:+.4f}" for x in comparison['improvements']['f1']]
    })
    
    print(df_comparison.to_string(index=False))
    
    avg_auc_improvement = np.mean(comparison['improvements']['auc'])
    avg_f1_improvement = np.mean(comparison['improvements']['f1'])
    
    print("=" * 80)
    print(f"Average AUC Improvement: {avg_auc_improvement:+.4f}")
    print(f"Average F1 Improvement: {avg_f1_improvement:+.4f}")
    
    return comparison


if __name__ == "__main__":
    # Example usage
    print("GANTGNN Result Analysis with Statistical Reporting")
    print("=" * 60)
    
    # Quick summary (if results file exists)
    summary = quick_performance_summary('all_results_physics_True.pt')
    
    # Full analysis example
    analyzer = ResultAnalyzer()
    
    # Load results
    physics_results = analyzer.load_results('all_results_physics_True.pt')
    baseline_results = analyzer.load_results('all_results_physics_False.pt')
    
    # Generate comprehensive report
    analyzer.generate_comprehensive_report(
        physics_results, 
        baseline_results,
        output_dir='./Results/comprehensive_analysis'
    )