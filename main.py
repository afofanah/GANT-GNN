import argparse
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from train import (
    run_experiment,
    run_all_experiments,
    run_statistical_experiments,
    generate_comprehensive_analysis,
    hyperparameter_optimization
)

from utils import (
    ResultAnalyzer, 
    quick_performance_summary, 
    compare_experiment_results
)

from datasets import DatasetManager, get_preprocessing_details, get_hyperparameter_grid


def create_config_from_args(args) -> Dict[str, Any]:
    config = {}
    
    if hasattr(args, 'hidden_dim') and args.hidden_dim:
        config['hidden_dim'] = args.hidden_dim
    if hasattr(args, 'output_dim') and args.output_dim:
        config['output_dim'] = args.output_dim
    if hasattr(args, 'num_gcn_layers') and args.num_gcn_layers:
        config['num_gcn_layers'] = args.num_gcn_layers
    if hasattr(args, 'num_attention_heads') and args.num_attention_heads:
        config['num_attention_heads'] = args.num_attention_heads
    if hasattr(args, 'dropout') and args.dropout is not None:
        config['dropout'] = args.dropout
    
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if hasattr(args, 'weight_decay') and args.weight_decay:
        config['weight_decay'] = args.weight_decay
    if hasattr(args, 'num_epochs') and args.num_epochs:
        config['num_epochs'] = args.num_epochs
    
    if hasattr(args, 'anomaly_ratio') and args.anomaly_ratio:
        config['anomaly_ratio'] = args.anomaly_ratio
    
    return config


def print_experiment_banner(mode: str, datasets: List[str], use_physics: bool, **kwargs):
    print(f"{mode.upper()}: {', '.join(datasets)} | Physics: {'ON' if use_physics else 'OFF'} | Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")


def validate_datasets(datasets: List[str]) -> List[str]:
    dataset_manager = DatasetManager()
    all_datasets = dataset_manager.get_all_datasets()
    
    valid_datasets = []
    for dataset in datasets:
        if dataset.lower() in [d.lower() for d in all_datasets]:
            valid_datasets.append(dataset.lower())
        else:
            print(f"Warning: Unknown dataset '{dataset}' - skipping")
    
    return valid_datasets


def run_single_experiment(args, config):
    dataset = args.dataset.lower()
    use_physics = args.use_physics and not args.no_physics
    
    print_experiment_banner('single', [dataset], use_physics, **config)
    
    result = run_experiment(
        dataset_name=dataset,
        use_physics=use_physics,
        output_dir=args.output_dir,
        **config
    )
    
    perf = result['report']['performance_metrics']['anomaly_detection']
    class_perf = result['report']['performance_metrics']['classification']
    
    print(f"{dataset.upper()}: AUC={perf['auc_roc']:.4f}, F1={perf['optimal_f1']:.4f}, Acc={class_perf['accuracy']:.4f}")
    
    return result


def run_statistical_mode(args, config):
    datasets = validate_datasets(args.datasets)
    if not datasets:
        print("No valid datasets provided")
        return None
    
    use_physics = args.use_physics and not args.no_physics
    
    print_experiment_banner('statistical', datasets, use_physics, 
                           num_runs=args.num_runs, **config)
    
    results = run_statistical_experiments(
        datasets=datasets,
        use_physics=use_physics,
        num_runs=args.num_runs,
        output_dir=args.output_dir,
        **config
    )
    
    print(f"Statistical results ({args.num_runs} runs):")
    for dataset, result in results.items():
        if 'stats' in result:
            stats = result['stats']
            print(f"  {dataset.upper()}: AUC={stats['auc']['mean']:.4f}±{stats['auc']['std']:.3f}, "
                  f"F1={stats['optimal_f1']['mean']:.4f}±{stats['optimal_f1']['std']:.3f}")
    
    return results


def run_hyperopt_mode(args, config):
    dataset = args.dataset.lower()
    use_physics = args.use_physics and not args.no_physics
    
    print_experiment_banner('hyperopt', [dataset], use_physics,
                           trials=args.trials, **config)
    
    results = hyperparameter_optimization(
        dataset_name=dataset,
        use_physics=use_physics,
        n_trials=args.trials
    )
    
    print(f"Hyperopt {dataset}: Best Score={results['best_score']:.4f}")
    print(f"Best params: {results['best_params']}")
    
    return results


def run_compare_mode(args, config):
    datasets = validate_datasets(args.datasets)
    if not datasets:
        print("No valid datasets provided")
        return None
    
    print_experiment_banner('compare', datasets, True, 
                           num_runs=args.num_runs, **config)
    
    print("Running physics-aware experiments...")
    physics_results = run_statistical_experiments(
        datasets=datasets,
        use_physics=True,
        num_runs=args.num_runs,
        output_dir=f"{args.output_dir}/physics_experiments",
        **config
    )
    
    print("Running baseline experiments...")
    baseline_results = run_statistical_experiments(
        datasets=datasets,
        use_physics=False,
        num_runs=args.num_runs,
        output_dir=f"{args.output_dir}/baseline_experiments",
        **config
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    physics_file = Path(args.output_dir) / f'physics_results_{timestamp}.json'
    baseline_file = Path(args.output_dir) / f'baseline_results_{timestamp}.json'
    
    with open(physics_file, 'w') as f:
        json.dump({k: {'stats': v['stats'], 'use_physics': v['use_physics']} 
                   for k, v in physics_results.items()}, f, indent=2, default=str)
    
    with open(baseline_file, 'w') as f:
        json.dump({k: {'stats': v['stats'], 'use_physics': v['use_physics']} 
                   for k, v in baseline_results.items()}, f, indent=2, default=str)
    
    comparison = compare_experiment_results(
        str(physics_file), str(baseline_file),
        labels=['Physics-Aware', 'Baseline']
    )
    
    print(f"Comparison completed. Results saved to: {args.output_dir}")
    
    return {'physics': physics_results, 'baseline': baseline_results, 'comparison': comparison}


def run_all_datasets_mode(args, config):
    dataset_manager = DatasetManager()
    
    all_datasets = dataset_manager.get_all_datasets()
    real_world_datasets = dataset_manager.get_real_world_datasets()
    
    datasets = real_world_datasets if args.real_world_only else all_datasets
    
    use_physics = args.use_physics and not args.no_physics
    
    print_experiment_banner('all-datasets', datasets, use_physics, **config)
    
    results, summary_df = run_all_experiments(
        datasets=datasets,
        use_physics=use_physics,
        output_dir=args.output_dir,
        **config
    )
    
    print(f"All datasets experiment completed. Physics: {use_physics}, Datasets: {len(datasets)}")
    
    quick_performance_summary(
        f"{args.output_dir}/all_results_{'physics' if use_physics else 'baseline'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt",
        print_details=True
    )
    
    return results


def generate_analysis_mode(args):
    physics_file = args.physics_results
    baseline_file = args.baseline_results if hasattr(args, 'baseline_results') else None
    
    print(f"Generating comprehensive analysis...")
    print(f"Physics results: {physics_file}")
    if baseline_file:
        print(f"Baseline results: {baseline_file}")
    
    generate_comprehensive_analysis(
        physics_results_file=physics_file,
        baseline_results_file=baseline_file,
        output_dir=args.output_dir
    )
    
    print(f"Analysis completed! Check {args.output_dir} for results.")


def display_dataset_info():
    dataset_manager = DatasetManager()
    categories = dataset_manager.get_dataset_categories()
    
    print("\nAvailable Datasets by Category:")
    print("="*50)
    
    for category, datasets in categories.items():
        print(f"\n{category}:")
        for dataset in datasets:
            print(f"  - {dataset}")
    
    print("\nReal-World Anomaly Detection Focus:")
    print("-" * 40)
    real_world = dataset_manager.get_real_world_datasets()
    for dataset in real_world:
        print(f"  - {dataset}")
    
    print("\nLarge-Scale Datasets (with mini-batch support):")
    print("-" * 40)
    large_scale = dataset_manager.get_large_scale_datasets()
    for dataset in large_scale:
        print(f"  - {dataset}")


def display_preprocessing_info():
    details = get_preprocessing_details()
    
    print("\nDataset Preprocessing Details:")
    print("="*50)
    
    for category, info in details.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        if isinstance(info, dict):
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {info}")


def display_hyperparameter_info():
    grid = get_hyperparameter_grid()
    
    print("\nHyperparameter Search Grid:")
    print("="*50)
    
    for category, params in grid.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for param, values in params.items():
            print(f"  {param}: {values}")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced GANTGNN Experiment Runner with Real-World Anomaly Focus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single experiment:     python main.py --mode single --dataset amazon --use-physics
  Statistical analysis:  python main.py --mode statistical --datasets amazon yelpchi credit-fraud --num-runs 5
  Hyperparameter opt:    python main.py --mode hyperopt --dataset amazon --trials 20
  Model comparison:      python main.py --mode compare --datasets amazon yelpchi --num-runs 3
  All datasets:          python main.py --mode all-datasets --real-world-only
  Generate analysis:     python main.py --mode analyze --physics-results results.pt
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'statistical', 'hyperopt', 'compare', 
                               'all-datasets', 'analyze', 'info'],
                       help='Experiment mode')
    
    parser.add_argument('--dataset', type=str, 
                       help='Single dataset for experiments')
    parser.add_argument('--datasets', nargs='*',
                       help='Multiple datasets for experiments')
    
    parser.add_argument('--output-dir', type=str, default='./Results', 
                       help='Output directory (default: ./Results)')
    parser.add_argument('--use-physics', action='store_true', default=True, 
                       help='Enable physics-aware components (default: True)')
    parser.add_argument('--no-physics', action='store_true', 
                       help='Disable physics-aware components')
    
    parser.add_argument('--hidden-dim', type=int, default=128,
                       choices=[64, 128, 256, 512])
    parser.add_argument('--output-dim', type=int, default=64,
                       choices=[32, 64, 128, 256])
    parser.add_argument('--num-gcn-layers', type=int, default=3,
                       choices=[2, 3, 4, 5])
    parser.add_argument('--num-attention-heads', type=int, default=4,
                       choices=[2, 4, 8, 16])
    parser.add_argument('--dropout', type=float, default=0.2,
                       choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       choices=[0.0001, 0.0005, 0.001, 0.005, 0.01])
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       choices=[1e-5, 1e-4, 1e-3, 1e-2])
    parser.add_argument('--num-epochs', type=int, default=200,
                       choices=[100, 150, 200, 250, 300])
    
    parser.add_argument('--anomaly-ratio', type=float, default=0.05,
                       choices=[0.03, 0.05, 0.07, 0.1])
    
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--real-world-only', action='store_true',
                       help='Use only real-world anomaly datasets')
    
    parser.add_argument('--physics-results', type=str,
                       help='Physics results file for analysis mode')
    parser.add_argument('--baseline-results', type=str,
                       help='Baseline results file for comparison')
    
    parser.add_argument('--info', type=str, 
                       choices=['datasets', 'preprocessing', 'hyperparameters'],
                       help='Display information about datasets, preprocessing, or hyperparameters')
    
    args = parser.parse_args()
    
    if args.mode == 'info' or args.info:
        if args.info == 'datasets' or args.mode == 'info':
            display_dataset_info()
        elif args.info == 'preprocessing':
            display_preprocessing_info()
        elif args.info == 'hyperparameters':
            display_hyperparameter_info()
        return
    
    if args.mode in ['single', 'hyperopt'] and not args.dataset:
        parser.error(f"Mode '{args.mode}' requires --dataset")
    
    if args.mode in ['statistical', 'compare'] and not args.datasets:
        parser.error(f"Mode '{args.mode}' requires --datasets")
    
    if args.mode == 'analyze' and not args.physics_results:
        parser.error("Mode 'analyze' requires --physics-results")
    
    config = create_config_from_args(args)
    
    if args.mode == 'single':
        run_single_experiment(args, config)
    
    elif args.mode == 'statistical':
        run_statistical_mode(args, config)
    
    elif args.mode == 'hyperopt':
        run_hyperopt_mode(args, config)
    
    elif args.mode == 'compare':
        run_compare_mode(args, config)
    
    elif args.mode == 'all-datasets':
        run_all_datasets_mode(args, config)
    
    elif args.mode == 'analyze':
        generate_analysis_mode(args)
    
    print(f"Experiments completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()