import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from datetime import datetime
import json
from pathlib import Path
import copy
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from models.model import create_enhanced_model
from datasets import OptimizedDatasetManager, create_batch_data
from utils import EarlyStopping, LearningRateScheduler, calculate_metrics


class OptimizedGANTGNNTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 lr=0.001, weight_decay=1e-4, use_scheduler=True):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        if use_scheduler:
            self.scheduler = LearningRateScheduler(self.optimizer, mode='plateau')
        else:
            self.scheduler = None
        
        self.early_stopping = EarlyStopping(patience=30, min_delta=0.001)
        
        self.history = {
            'train_loss': [],
            'val_metrics': [],
            'learning_rates': [],
            'loss_components': []
        }
        
        self.val_loader = None
        self.best_model_state = None
        
    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        batch_count = 0
        epoch_loss_components = defaultdict(list)
        
        for batch in data_loader:
            self.optimizer.zero_grad()
            
            features = batch['features'].to(self.device)
            adj_matrix = batch['adj_matrix'].to(self.device)
            labels = batch['labels'].to(self.device) if batch['labels'] is not None else None
            anomaly_labels = batch['anomaly_labels'].to(self.device) if batch['anomaly_labels'] is not None else None
            
            outputs = self.model(features, adj_matrix, labels, anomaly_labels)
            
            if 'loss' in outputs:
                loss = outputs['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                if 'loss_components' in outputs:
                    for component, value in outputs['loss_components'].items():
                        epoch_loss_components[component].append(value)
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_loss_components = {component: np.mean(values) if values else 0 for component, values in epoch_loss_components.items()}
        
        return avg_loss, avg_loss_components
    
    def evaluate(self, data_loader):
        self.model.eval()
        all_scores = []
        all_predictions = []
        all_labels = []
        all_anomaly_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                adj_matrix = batch['adj_matrix'].to(self.device)
                labels = batch['labels'].to(self.device) if batch['labels'] is not None else None
                anomaly_labels = batch['anomaly_labels'].to(self.device) if batch['anomaly_labels'] is not None else None
                
                outputs = self.model(features, adj_matrix, labels, anomaly_labels)
                
                if 'logits' in outputs and labels is not None:
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                    all_predictions.extend(predictions.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
                
                if 'anomaly_scores' in outputs and anomaly_labels is not None:
                    scores = outputs['anomaly_scores']
                    all_scores.extend(scores.cpu().numpy().flatten())
                    all_anomaly_labels.extend(anomaly_labels.cpu().numpy().flatten())
        
        metrics = {}
        
        if all_labels and all_predictions:
            metrics['classification_accuracy'] = accuracy_score(all_labels, all_predictions)
            metrics['classification_f1'] = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        if all_anomaly_labels and all_scores:
            all_anomaly_labels = np.array(all_anomaly_labels)
            all_scores = np.array(all_scores)
            
            if len(np.unique(all_anomaly_labels)) > 1:
                metrics['anomaly_auc'] = roc_auc_score(all_anomaly_labels, all_scores)
                
                threshold = 0.5
                predictions = (all_scores > threshold).astype(int)
                metrics['anomaly_f1'] = f1_score(all_anomaly_labels, predictions, zero_division=0)
                metrics['anomaly_precision'] = precision_score(all_anomaly_labels, predictions, zero_division=0)
                metrics['anomaly_recall'] = recall_score(all_anomaly_labels, predictions, zero_division=0)
                
                from sklearn.metrics import precision_recall_curve
                precision, recall, thresholds = precision_recall_curve(all_anomaly_labels, all_scores)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                optimal_f1 = np.max(f1_scores)
                metrics['optimal_f1'] = optimal_f1
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs=200, verbose=True):
        self.val_loader = val_loader
        best_f1 = 0
        
        for epoch in range(num_epochs):
            train_loss, loss_components = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_metrics'].append(val_metrics)
            self.history['loss_components'].append(loss_components)
            
            if self.scheduler:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
                
                metric_to_track = val_metrics.get('optimal_f1', val_metrics.get('anomaly_f1', 0))
                self.scheduler.step(metric_to_track)
            
            current_f1 = val_metrics.get('optimal_f1', val_metrics.get('anomaly_f1', 0))
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, "
                      f"AUC: {val_metrics.get('anomaly_auc', 0):.4f}, "
                      f"F1: {current_f1:.4f}")
            
            if self.early_stopping(current_f1, self.model):
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def evaluate_comprehensive(self, data_loader):
        self.model.eval()
        all_scores = []
        all_predictions = []
        all_labels = []
        all_anomaly_labels = []
        all_anomaly_scores = []
        
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                adj_matrix = batch['adj_matrix'].to(self.device)
                labels = batch['labels'].to(self.device) if batch['labels'] is not None else None
                anomaly_labels = batch['anomaly_labels'].to(self.device) if batch['anomaly_labels'] is not None else None
                
                results = self.model.detect_anomalies(features, adj_matrix, validation_labels=anomaly_labels)
                
                if anomaly_labels is not None:
                    all_anomaly_scores.extend(results['fused_scores'].cpu().numpy())
                    all_anomaly_labels.extend(anomaly_labels.cpu().numpy().flatten())
                    all_predictions.extend(results['predictions'].cpu().numpy())
                
                if labels is not None:
                    outputs = self.model(features, adj_matrix)
                    if 'logits' in outputs:
                        class_predictions = torch.argmax(outputs['logits'], dim=-1)
                        all_labels.extend(labels.cpu().numpy().flatten())
                        all_scores.extend(class_predictions.cpu().numpy().flatten())
        
        comprehensive_metrics = {}
        
        if all_anomaly_labels and all_anomaly_scores:
            all_anomaly_labels = np.array(all_anomaly_labels)
            all_anomaly_scores = np.array(all_anomaly_scores)
            all_predictions = np.array(all_predictions)
            
            comprehensive_metrics.update({
                'anomaly_scores': all_anomaly_scores,
                'anomaly_predictions': all_predictions,
                'anomaly_true_labels': all_anomaly_labels
            })
            
            if len(np.unique(all_anomaly_labels)) > 1:
                comprehensive_metrics['anomaly_auc'] = roc_auc_score(all_anomaly_labels, all_anomaly_scores)
                
                from sklearn.metrics import average_precision_score
                comprehensive_metrics['anomaly_ap'] = average_precision_score(all_anomaly_labels, all_anomaly_scores)
        
        if all_labels and all_scores:
            comprehensive_metrics.update({
                'classification_predictions': np.array(all_scores),
                'classification_true_labels': np.array(all_labels)
            })
        
        return comprehensive_metrics


def create_data_loaders(data, metadata, use_minibatch=False, batch_size=1024, train_loader=None):
    if use_minibatch and train_loader is not None:
        val_data = create_batch_data(data)
        val_loader = [val_data]
        return train_loader, val_loader
    
    train_data = create_batch_data(data)
    val_data = create_batch_data(data)
    
    train_loader = [train_data]
    val_loader = [val_data]
    
    return train_loader, val_loader


def run_optimized_experiment(dataset_name: str, use_physics: bool = True, anomaly_ratio: float = 0.05,
                  num_epochs: int = 200, lr: float = 0.001, hidden_dim: int = 128,
                  output_dim: int = 64, num_gcn_layers: int = 3, num_attention_heads: int = 4,
                  dropout: float = 0.2, weight_decay: float = 1e-4, output_dir: str = './Results') -> Dict[str, Any]:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset_manager = OptimizedDatasetManager()
    
    use_minibatch = dataset_name.lower() in ['ogbn-arxiv', 'reddit']
    
    data, metadata, minibatch_loader = dataset_manager.load_dataset(
        dataset_name, use_minibatch=use_minibatch, batch_size=1024
    )
    
    train_loader, val_loader = create_data_loaders(
        data, metadata, use_minibatch, 1024, minibatch_loader
    )
    
    num_classes = metadata.get('num_classes', 2)
    
    model = create_enhanced_model(
        input_dim=data.x.shape[1],
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_gcn_layers=num_gcn_layers,
        num_classes=num_classes,
        dropout=dropout,
        num_attention_heads=num_attention_heads,
        use_physics=use_physics
    )
    
    trainer = OptimizedGANTGNNTrainer(model, device, lr, weight_decay)
    
    history = trainer.train(train_loader, val_loader, num_epochs, verbose=True)
    
    test_metrics = trainer.evaluate_comprehensive(val_loader)
    
    anomaly_metrics = {}
    classification_metrics = {}
    
    if 'anomaly_scores' in test_metrics:
        y_true = test_metrics['anomaly_true_labels']
        y_scores = test_metrics['anomaly_scores']
        y_pred = test_metrics['anomaly_predictions']
        
        anomaly_metrics = calculate_metrics(y_true, y_pred, y_scores)
    
    if 'classification_predictions' in test_metrics:
        y_true_class = test_metrics['classification_true_labels']
        y_pred_class = test_metrics['classification_predictions']
        y_scores_class = y_pred_class
        
        classification_metrics = calculate_metrics(y_true_class, y_pred_class, y_scores_class)
    
    results = {
        'dataset': dataset_name,
        'metadata': metadata,
        'model': model,
        'trainer': trainer,
        'history': history,
        'test_metrics': test_metrics,
        'use_physics': use_physics,
        'hyperparameters': {
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'num_gcn_layers': num_gcn_layers,
            'num_attention_heads': num_attention_heads,
            'dropout': dropout,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs
        },
        'report': {
            'performance_metrics': {
                'anomaly_detection': anomaly_metrics,
                'classification': classification_metrics
            },
            'dataset_info': metadata,
            'training_info': {
                'total_epochs': len(history['train_loss']),
                'final_loss': history['train_loss'][-1] if history['train_loss'] else 0,
                'best_f1': max([m.get('optimal_f1', 0) for m in history['val_metrics']]) if history['val_metrics'] else 0
            }
        }
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    physics_str = "physics" if use_physics else "baseline"
    result_file = output_path / f'{dataset_name}_{physics_str}_{timestamp}.pt'
    
    torch.save(results, result_file)
    
    return results


def run_statistical_experiments_optimized(datasets: List[str], use_physics: bool = True, anomaly_ratio: float = 0.05,
                               num_runs: int = 5, output_dir: str = './Results', **kwargs) -> Dict[str, Any]:
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\nRunning {num_runs} experiments on {dataset}...")
        dataset_results = []
        
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")
            
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)
            
            result = run_optimized_experiment(
                dataset_name=dataset,
                use_physics=use_physics,
                anomaly_ratio=anomaly_ratio,
                output_dir=f"{output_dir}/statistical_runs/{dataset}/run_{run+1}",
                **kwargs
            )
            
            dataset_results.append(result)
        
        metrics_list = []
        for result in dataset_results:
            anom_metrics = result['report']['performance_metrics']['anomaly_detection']
            class_metrics = result['report']['performance_metrics']['classification']
            
            run_metrics = {
                'auc': anom_metrics.get('auc_roc', 0),
                'f1': anom_metrics.get('f1_score', 0),
                'optimal_f1': anom_metrics.get('optimal_f1', 0),
                'precision': anom_metrics.get('precision', 0),
                'recall': anom_metrics.get('recall', 0),
                'accuracy': class_metrics.get('accuracy', 0)
            }
            metrics_list.append(run_metrics)
        
        stats = {}
        for metric in ['auc', 'f1', 'optimal_f1', 'precision', 'recall', 'accuracy']:
            values = [m[metric] for m in metrics_list]
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        all_results[dataset] = {
            'individual_results': dataset_results,
            'stats': stats,
            'use_physics': use_physics,
            'num_runs': num_runs,
            'dataset_info': dataset_results[0]['metadata'] if dataset_results else {}
        }
        
        print(f"  {dataset}: AUC={stats['auc']['mean']:.4f}±{stats['auc']['std']:.3f}, "
              f"F1={stats['optimal_f1']['mean']:.4f}±{stats['optimal_f1']['std']:.3f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    physics_str = "physics" if use_physics else "baseline"
    
    stats_file = Path(output_dir) / f'statistical_results_{physics_str}_{timestamp}.json'
    
    serializable_results = {}
    for dataset, result in all_results.items():
        serializable_results[dataset] = {
            'stats': result['stats'],
            'use_physics': result['use_physics'],
            'num_runs': result['num_runs'],
            'dataset_info': result['dataset_info']
        }
    
    with open(stats_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\nStatistical results saved to: {stats_file}")
    
    return all_results


def run_all_experiments_optimized(datasets: List[str], use_physics: bool = True, anomaly_ratio: float = 0.05,
                       output_dir: str = './Results', **kwargs) -> Tuple[Dict[str, Any], pd.DataFrame]:
    
    all_results = {}
    summary_data = []
    
    for dataset in datasets:
        print(f"\nRunning experiment on {dataset}...")
        
        result = run_optimized_experiment(
            dataset_name=dataset,
            use_physics=use_physics,
            anomaly_ratio=anomaly_ratio,
            output_dir=f"{output_dir}/individual_experiments/{dataset}",
            **kwargs
        )
        
        all_results[dataset] = result
        
        anom_metrics = result['report']['performance_metrics']['anomaly_detection']
        class_metrics = result['report']['performance_metrics']['classification']
        
        summary_row = {
            'Dataset': dataset.capitalize(),
            'Physics': use_physics,
            'Nodes': result['metadata']['num_nodes'],
            'Edges': result['metadata']['num_edges'],
            'Features': result['metadata']['num_features'],
            'Anomalies': result['metadata']['num_anomalies'],
            'AUC': anom_metrics.get('auc_roc', 0),
            'F1': anom_metrics.get('f1_score', 0),
            'Optimal_F1': anom_metrics.get('optimal_f1', 0),
            'Precision': anom_metrics.get('precision', 0),
            'Recall': anom_metrics.get('recall', 0),
            'Accuracy': class_metrics.get('accuracy', 0)
        }
        
        summary_data.append(summary_row)
        
        print(f"  {dataset}: AUC={anom_metrics.get('auc_roc', 0):.4f}, "
              f"F1={anom_metrics.get('optimal_f1', 0):.4f}, "
              f"Acc={class_metrics.get('accuracy', 0):.4f}")
    
    summary_df = pd.DataFrame(summary_data)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    physics_str = "physics" if use_physics else "baseline"
    
    results_file = Path(output_dir) / f'all_results_{physics_str}_{timestamp}.pt'
    summary_file = Path(output_dir) / f'summary_{physics_str}_{timestamp}.csv'
    
    torch.save(all_results, results_file)
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    
    return all_results, summary_df


def hyperparameter_optimization_optimized(dataset_name: str, use_physics: bool = True, n_trials: int = 20):
    from datasets import get_hyperparameter_grid
    
    param_grid = get_hyperparameter_grid()
    
    best_score = 0
    best_params = {}
    all_trials = []
    
    for trial in range(n_trials):
        params = {}
        for category, param_dict in param_grid.items():
            for param, values in param_dict.items():
                params[param] = np.random.choice(values)
        
        result = run_optimized_experiment(
            dataset_name=dataset_name,
            use_physics=use_physics,
            **params
        )
        
        score = result['report']['performance_metrics']['anomaly_detection'].get('optimal_f1', 0)
        
        trial_result = {
            'trial': trial + 1,
            'params': params,
            'score': score,
            'auc': result['report']['performance_metrics']['anomaly_detection'].get('auc_roc', 0)
        }
        
        all_trials.append(trial_result)
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
        
        print(f"Trial {trial+1}/{n_trials}: Score={score:.4f}")
    
    optimization_results = {
        'dataset': dataset_name,
        'best_score': best_score,
        'best_params': best_params,
        'all_trials': all_trials
    }
    
    return optimization_results


def batch_run_complex_datasets(use_physics: bool = True, num_runs: int = 3, output_dir: str = './Results'):
    complex_datasets = ['yelpchi', 'social-bot', 't-finance', 't-social', 'amazon', 'reddit']
    
    optimized_params = {
        'hidden_dim': 256,
        'output_dim': 128,
        'num_gcn_layers': 4,
        'num_attention_heads': 8,
        'dropout': 0.2,
        'lr': 0.0005,
        'weight_decay': 1e-4,
        'num_epochs': 150
    }
    
    print(f"Running optimized experiments on complex datasets with physics={use_physics}")
    
    results = run_statistical_experiments_optimized(
        datasets=complex_datasets,
        use_physics=use_physics,
        num_runs=num_runs,
        output_dir=output_dir,
        **optimized_params
    )
    
    return results


def memory_efficient_training(dataset_name: str, use_physics: bool = True, 
                             max_memory_mb: int = 8192):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        
        if total_memory < max_memory_mb:
            batch_size = 512
            hidden_dim = 128
            num_gcn_layers = 3
        else:
            batch_size = 1024
            hidden_dim = 256
            num_gcn_layers = 4
    else:
        batch_size = 256
        hidden_dim = 64
        num_gcn_layers = 2
    
    result = run_optimized_experiment(
        dataset_name=dataset_name,
        use_physics=use_physics,
        hidden_dim=hidden_dim,
        num_gcn_layers=num_gcn_layers,
        dropout=0.3,
        lr=0.001,
        num_epochs=100
    )
    
    if device.type == 'cuda':
        memory_usage = torch.cuda.max_memory_allocated() // (1024**2)
        print(f"Peak memory usage: {memory_usage}MB")
    
    return result