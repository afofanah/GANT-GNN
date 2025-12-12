import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, f1_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import networkx as nx
from collections import defaultdict
import json
import os
from pathlib import Path

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0
})


class OptimizedResultAnalyzer:
    def __init__(self, results_dict=None, output_dir='./Results'):
        self.results = results_dict or {}
        self.comparison_data = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('default')
        self.colors = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6', '#E67E22', '#1ABC9C']
        
    def load_results(self, filepath):
        if isinstance(filepath, str):
            self.results = torch.load(filepath, map_location='cpu')
        else:
            self.results = filepath
        return self.results
    
    def plot_statistical_comparison(self, results_data, save_path=None):
        datasets = list(results_data.keys())
        metrics = ['auc', 'f1', 'optimal_f1', 'precision', 'recall', 'accuracy']
        metric_names = ['AUC-ROC', 'F1 Score', 'Optimal F1', 'Precision', 'Recall', 'Accuracy']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Enhanced GANTGNN: Statistical Performance Analysis', fontsize=16, fontweight='bold')

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
                
                for j, (mean, std) in enumerate(zip(means, stds)):
                    ax.text(j, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_curves_fast(self, results, dataset_name, save_path=None):
        if dataset_name not in results:
            return
        
        trainer = results[dataset_name].get('trainer')
        if not trainer or not hasattr(trainer, 'history'):
            return
            
        history = trainer.history
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Training Curves - {dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        
        if 'train_loss' in history and history['train_loss']:
            axes[0, 0].plot(history['train_loss'], color='#2E86C1', linewidth=2)
            axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        if 'val_metrics' in history and history['val_metrics']:
            epochs = range(len(history['val_metrics']))
            val_aucs = [m.get('anomaly_auc', 0) for m in history['val_metrics']]
            val_f1s = [m.get('anomaly_f1', 0) for m in history['val_metrics']]
            
            axes[0, 1].plot(epochs, val_aucs, label='AUC', color='#E74C3C', linewidth=2)
            axes[0, 1].plot(epochs, val_f1s, label='F1', color='#27AE60', linewidth=2)
            axes[0, 1].set_title('Validation Metrics')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        if 'learning_rates' in history and history['learning_rates']:
            axes[1, 0].plot(history['learning_rates'], color='#F39C12', linewidth=2)
            axes[1, 0].set_title('Learning Rate Evolution')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'val_metrics' in history and history['val_metrics']:
            val_accuracies = [m.get('classification_accuracy', 0) for m in history['val_metrics']]
            if any(acc > 0 for acc in val_accuracies):
                epochs = range(len(val_accuracies))
                axes[1, 1].plot(epochs, val_accuracies, color='#16A085', linewidth=2)
                axes[1, 1].set_title('Classification Accuracy')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Accuracy')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_pr_curves(self, results, dataset_name, save_path=None):
        if dataset_name not in results:
            return
        
        trainer = results[dataset_name].get('trainer')
        if not trainer:
            return
        
        test_metrics = trainer.evaluate_comprehensive(trainer.val_loader)
            
        if 'anomaly_scores' not in test_metrics:
            return
        
        y_true = test_metrics['anomaly_true_labels']
        y_scores = test_metrics['anomaly_scores']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'ROC and Precision-Recall Curves - {dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = test_metrics['anomaly_auc']
        
        axes[0].plot(fpr, tpr, color='#E74C3C', lw=3, label=f'ROC curve (AUC = {auc_score:.3f})')
        axes[0].plot([0, 1], [0, 1], color='#34495E', lw=2, linestyle='--', label='Random')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap_score = test_metrics['anomaly_ap']
        
        axes[1].plot(recall, precision, color='#27AE60', lw=3, label=f'PR curve (AP = {ap_score:.3f})')
        axes[1].axhline(y=np.mean(y_true), color='#34495E', linestyle='--', label='Random')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend(loc="lower left")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, results, dataset_name, save_path=None):
        if dataset_name not in results:
            return
        
        trainer = results[dataset_name].get('trainer')
        if not trainer:
            return
        
        test_metrics = trainer.evaluate_comprehensive(trainer.val_loader)
            
        if 'anomaly_predictions' not in test_metrics:
            return
        
        y_true = np.asarray(test_metrics['anomaly_true_labels']).flatten().astype(int)
        y_pred = np.asarray(test_metrics['anomaly_predictions']).flatten().astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'shrink': 0.8},
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title(f'Confusion Matrix - {dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        total = np.sum(cm)
        if total > 0:
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j + 0.5, i + 0.7, f'({cm[i,j]/total*100:.1f}%)', 
                            ha='center', va='center', fontsize=11, color='red', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_embedding_analysis_fast(self, results, dataset_name, save_path=None):
        if dataset_name not in results:
            return

        trainer = results[dataset_name].get('trainer')
        model = results[dataset_name].get('model')
        
        if not trainer or not model:
            return

        if not hasattr(trainer, 'val_loader') or trainer.val_loader is None:
            return

        sample_batch = next(iter(trainer.val_loader))

        features = sample_batch['features'].to(trainer.device)
        adj_matrix = sample_batch['adj_matrix'].to(trainer.device)
        labels = sample_batch['labels']
        anomaly_labels = sample_batch.get('anomaly_labels')
        
        if anomaly_labels is None:
            return
        
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(anomaly_labels, torch.Tensor):
            anomaly_labels = anomaly_labels.cpu().numpy()
            
        labels = np.asarray(labels).flatten().astype(int)
        anomaly_labels = np.asarray(anomaly_labels).flatten().astype(int)
        
        with torch.no_grad():
            model.eval()
            outputs = model(features, adj_matrix)
            
            embeddings = None
            if isinstance(outputs, dict):
                if 'final_embeddings' in outputs:
                    embeddings = outputs['final_embeddings']
                elif 'gcn_embeddings' in outputs:
                    embeddings = outputs['gcn_embeddings']
                elif 'embeddings' in outputs:
                    embeddings = outputs['embeddings']
            
            if embeddings is None:
                embeddings = features
            
            embeddings = embeddings.cpu().numpy().squeeze()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Embedding Analysis - {dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        
        if embeddings.shape[0] > 50 and embeddings.shape[1] > 1:
            perplexity = min(30, max(5, embeddings.shape[0] // 4))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            scatter = axes[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                       c=labels, cmap='tab10', alpha=0.7, s=20)
            axes[0, 0].set_title('t-SNE by Node Classes')
            axes[0, 0].set_xlabel('t-SNE 1')
            axes[0, 0].set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=axes[0, 0])
            
            color_array = np.where(anomaly_labels == 0, '#3498DB', '#E74C3C')
            axes[0, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=color_array, alpha=0.7, s=20)
            axes[0, 1].set_title('t-SNE by Anomaly Status')
            axes[0, 1].set_xlabel('t-SNE 1')
            axes[0, 1].set_ylabel('t-SNE 2')
        
        if embeddings.shape[1] > 1:
            pca = PCA()
            embeddings_pca = pca.fit_transform(embeddings)
            
            axes[0, 2].plot(np.cumsum(pca.explained_variance_ratio_), 'bo-', linewidth=2, markersize=4)
            axes[0, 2].set_title('PCA Explained Variance')
            axes[0, 2].set_xlabel('Component')
            axes[0, 2].set_ylabel('Cumulative Explained Variance')
            axes[0, 2].grid(True, alpha=0.3)
        
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        normal_indices = np.where(anomaly_labels == 0)[0]
        anomaly_indices = np.where(anomaly_labels == 1)[0]
        
        normal_norms = embedding_norms[normal_indices]
        anomaly_norms = embedding_norms[anomaly_indices]
        
        if len(normal_norms) > 0:
            axes[1, 0].hist(normal_norms, bins=20, alpha=0.7, label='Normal', color='#3498DB')
        if len(anomaly_norms) > 0:
            axes[1, 0].hist(anomaly_norms, bins=20, alpha=0.7, label='Anomaly', color='#E74C3C')
        axes[1, 0].set_title('Embedding Magnitude Distribution')
        axes[1, 0].set_xlabel('L2 Norm')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        if embeddings.shape[0] > 10:
            n_clusters = min(5, len(np.unique(labels)))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                
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
                    axes[1, 1].set_title('Cluster Anomaly Ratio')
                    axes[1, 1].set_xlabel('Cluster ID')
                    axes[1, 1].set_ylabel('Anomaly Ratio')
                    axes[1, 1].grid(True, alpha=0.3)
        
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
                
                normal_sim_vals = normal_sim[np.triu_indices_from(normal_sim, k=1)]
                anomaly_sim_vals = anomaly_sim[np.triu_indices_from(anomaly_sim, k=1)]
                cross_sim_vals = cross_sim.flatten()
                
                sim_data = [normal_sim_vals, anomaly_sim_vals, cross_sim_vals]
                
                if all(len(data) > 0 for data in sim_data):
                    box_plot = axes[1, 2].boxplot(sim_data, labels=['Normal-Normal', 'Anomaly-Anomaly', 'Normal-Anomaly'])
                    axes[1, 2].set_title('Embedding Similarity Analysis')
                    axes[1, 2].set_ylabel('Cosine Similarity')
                    axes[1, 2].grid(True, alpha=0.3)
                    plt.setp(axes[1, 2].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comparison(self, comparison_data, save_path=None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Enhanced GANTGNN: Physics-Aware vs Standard Model Comparison', fontsize=16, fontweight='bold')
        
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
            
            ax.set_xlabel('Datasets')
            ax.set_ylabel(name)
            ax.set_title(f'{name} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([d.capitalize() for d in datasets], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        if len(metrics) % 3 != 0:
            axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_improvement_heatmap(self, comparison_data, save_path=None):
        datasets = comparison_data['datasets']
        metrics = ['auc', 'f1', 'precision', 'recall', 'accuracy']
        metric_names = ['AUC', 'F1', 'Precision', 'Recall', 'Accuracy']
        
        improvement_matrix = []
        for dataset in datasets:
            row = []
            for metric in metrics:
                idx = comparison_data['datasets'].index(dataset)
                improvement = comparison_data['improvements'][metric][idx]
                row.append(improvement)
            improvement_matrix.append(row)
        
        improvement_matrix = np.array(improvement_matrix)
        
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
        plt.xlabel('Metrics')
        plt.ylabel('Datasets')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        return improvement_matrix
    
    def compare_models(self, results_with_physics, results_without_physics):
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
            
            anom_with = with_physics['anomaly_detection']
            anom_without = without_physics['anomaly_detection']
            class_with = with_physics['classification']
            class_without = without_physics['classification']
            
            comparison['physics_enabled']['auc'].append(anom_with['auc_roc'])
            comparison['physics_enabled']['f1'].append(anom_with['optimal_f1'])
            comparison['physics_enabled']['precision'].append(anom_with['precision'])
            comparison['physics_enabled']['recall'].append(anom_with['recall'])
            comparison['physics_enabled']['accuracy'].append(class_with['accuracy'])
            
            comparison['physics_disabled']['auc'].append(anom_without['auc_roc'])
            comparison['physics_disabled']['f1'].append(anom_without['optimal_f1'])
            comparison['physics_disabled']['precision'].append(anom_without['precision'])
            comparison['physics_disabled']['recall'].append(anom_without['recall'])
            comparison['physics_disabled']['accuracy'].append(class_without['accuracy'])
            
            comparison['improvements']['auc'].append(anom_with['auc_roc'] - anom_without['auc_roc'])
            comparison['improvements']['f1'].append(anom_with['optimal_f1'] - anom_without['optimal_f1'])
            comparison['improvements']['precision'].append(anom_with['precision'] - anom_without['precision'])
            comparison['improvements']['recall'].append(anom_with['recall'] - anom_without['recall'])
            comparison['improvements']['accuracy'].append(class_with['accuracy'] - class_without['accuracy'])
        
        return comparison
    
    def generate_comprehensive_report(self, results_with_physics, results_without_physics=None, 
                                    hyperopt_results=None, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir / 'comprehensive_reports'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset in results_with_physics.keys():
            dataset_dir = output_dir / dataset
            dataset_dir.mkdir(exist_ok=True)
            
            self.plot_training_curves_fast(results_with_physics, dataset, 
                                    dataset_dir / 'training_curves.pdf')
            
            self.plot_embedding_analysis_fast(results_with_physics, dataset,
                                        dataset_dir / 'embedding_analysis.pdf')
            
            self.plot_roc_pr_curves(results_with_physics, dataset,
                                  dataset_dir / 'roc_pr_curves.pdf')
            
            self.plot_confusion_matrix(results_with_physics, dataset,
                                     dataset_dir / 'confusion_matrix.pdf')
        
        if results_without_physics:
            comparison = self.compare_models(results_with_physics, results_without_physics)
            self.plot_comparison(comparison, output_dir / 'model_comparison.pdf')
            self.plot_improvement_heatmap(comparison, output_dir / 'improvement_heatmap.pdf')


def quick_performance_summary(results_file, print_details=True):
    if isinstance(results_file, str):
        if results_file.endswith('.json'):
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
    
    analyzer = OptimizedResultAnalyzer()
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


class EarlyStopping:
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


def calculate_metrics(y_true, y_pred, y_scores):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_scores = np.asarray(y_scores).flatten()
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
    }
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_f1 = np.max(f1_scores)
    optimal_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
    
    metrics.update({
        'optimal_f1': optimal_f1,
        'optimal_threshold': optimal_threshold
    })
    
    return metrics