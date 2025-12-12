import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from datasets import DatasetManager
from train import run_experiment


class QuickExperiments:
    def __init__(self, output_dir='./Results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_real_world_evaluation(self, use_physics=True, num_runs=3):
        dataset_manager = DatasetManager()
        real_world_datasets = dataset_manager.get_real_world_datasets()
        
        results = {}
        print(f"Running real-world evaluation (Physics: {use_physics})...")
        
        for dataset in real_world_datasets:
            print(f"  {dataset}...")
            dataset_results = []
            
            for run in range(num_runs):
                torch.manual_seed(42 + run)
                np.random.seed(42 + run)
                
                result = run_experiment(
                    dataset_name=dataset,
                    use_physics=use_physics,
                    output_dir=str(self.output_dir / f'{dataset}_run_{run+1}')
                )
                dataset_results.append(result)
            
            anom_scores = [r['report']['performance_metrics']['anomaly_detection']['optimal_f1'] 
                          for r in dataset_results]
            
            results[dataset] = {
                'mean_f1': np.mean(anom_scores),
                'std_f1': np.std(anom_scores),
                'results': dataset_results
            }
            
            print(f"    F1: {results[dataset]['mean_f1']:.4f}±{results[dataset]['std_f1']:.3f}")
        
        return results
    
    def compare_physics_vs_baseline(self, datasets=None, num_runs=3):
        if datasets is None:
            dataset_manager = DatasetManager()
            datasets = dataset_manager.get_real_world_datasets()[:3]
        
        print(f"Comparing Physics vs Baseline on {datasets}...")
        
        physics_results = {}
        baseline_results = {}
        
        for dataset in datasets:
            print(f"  {dataset}...")
            
            print("    Physics...")
            physics_scores = []
            for run in range(num_runs):
                result = run_experiment(dataset, use_physics=True)
                physics_scores.append(result['report']['performance_metrics']['anomaly_detection']['optimal_f1'])
            physics_results[dataset] = {'mean': np.mean(physics_scores), 'std': np.std(physics_scores)}
            
            print("    Baseline...")
            baseline_scores = []
            for run in range(num_runs):
                result = run_experiment(dataset, use_physics=False)
                baseline_scores.append(result['report']['performance_metrics']['anomaly_detection']['optimal_f1'])
            baseline_results[dataset] = {'mean': np.mean(baseline_scores), 'std': np.std(baseline_scores)}
            
            improvement = physics_results[dataset]['mean'] - baseline_results[dataset]['mean']
            print(f"    Physics: {physics_results[dataset]['mean']:.4f}±{physics_results[dataset]['std']:.3f}")
            print(f"    Baseline: {baseline_results[dataset]['mean']:.4f}±{baseline_results[dataset]['std']:.3f}")
            print(f"    Improvement: {improvement:+.4f}")
        
        return {'physics': physics_results, 'baseline': baseline_results}
    
    def quick_single_run(self, dataset='amazon', use_physics=True):
        print(f"Quick run: {dataset} (Physics: {use_physics})")
        
        result = run_experiment(
            dataset_name=dataset,
            use_physics=use_physics,
            num_epochs=100,
            output_dir=str(self.output_dir / f'quick_{dataset}')
        )
        
        perf = result['report']['performance_metrics']['anomaly_detection']
        class_perf = result['report']['performance_metrics']['classification']
        
        print(f"Results: AUC={perf['auc_roc']:.4f}, F1={perf['optimal_f1']:.4f}, Acc={class_perf['accuracy']:.4f}")
        
        return result
    
    def hyperparameter_search(self, dataset='amazon', n_trials=10):
        from train import hyperparameter_optimization
        
        print(f"Hyperparameter search: {dataset} ({n_trials} trials)")
        
        results = hyperparameter_optimization(
            dataset_name=dataset,
            use_physics=True,
            n_trials=n_trials
        )
        
        print(f"Best score: {results['best_score']:.4f}")
        print(f"Best params: {results['best_params']}")
        
        return results
    
    def large_scale_test(self):
        print("Large-scale dataset test (OGBN-Arxiv)...")
        
        result = run_experiment(
            dataset_name='ogbn-arxiv',
            use_physics=True,
            num_epochs=50,
            output_dir=str(self.output_dir / 'large_scale_test')
        )
        
        perf = result['report']['performance_metrics']['anomaly_detection']
        print(f"Large-scale results: AUC={perf['auc_roc']:.4f}, F1={perf['optimal_f1']:.4f}")
        
        return result


def run_comprehensive_evaluation():
    exp = QuickExperiments()
    
    print("=== Comprehensive GANTGNN Evaluation ===")
    
    print("\n1. Real-world evaluation (Physics-aware)")
    physics_eval = exp.run_real_world_evaluation(use_physics=True, num_runs=3)
    
    print("\n2. Real-world evaluation (Baseline)")
    baseline_eval = exp.run_real_world_evaluation(use_physics=False, num_runs=3)
    
    print("\n3. Physics vs Baseline comparison")
    comparison = exp.compare_physics_vs_baseline(['amazon', 'yelpchi'], num_runs=3)
    
    print("\n4. Large-scale test")
    large_scale = exp.large_scale_test()
    
    print("\n=== Evaluation Complete ===")
    
    return {
        'physics_eval': physics_eval,
        'baseline_eval': baseline_eval,
        'comparison': comparison,
        'large_scale': large_scale
    }


def run_paper_experiments():
    exp = QuickExperiments()
    
    print("=== Paper Experiments Reproduction ===")
    
    dataset_manager = DatasetManager()
    real_world_datasets = dataset_manager.get_real_world_datasets()
    synthetic_datasets = dataset_manager.get_synthetic_datasets()
    
    all_results = {}
    
    print("\n1. Real-world anomaly detection datasets")
    for dataset in real_world_datasets:
        print(f"  Running {dataset}...")
        result = exp.quick_single_run(dataset, use_physics=True)
        all_results[f"{dataset}_physics"] = result
        
        result = exp.quick_single_run(dataset, use_physics=False)
        all_results[f"{dataset}_baseline"] = result
    
    print("\n2. Synthetic datasets")
    for dataset in synthetic_datasets[:3]:
        print(f"  Running {dataset}...")
        result = exp.quick_single_run(dataset, use_physics=True)
        all_results[f"{dataset}_physics"] = result
    
    print("\n3. Large-scale dataset")
    large_result = exp.large_scale_test()
    all_results['large_scale'] = large_result
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = exp.output_dir / f'paper_experiments_{timestamp}.pt'
    torch.save(all_results, results_file)
    
    print(f"\nPaper experiments complete. Results saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick GANTGNN Experiments')
    parser.add_argument('--mode', type=str, default='comprehensive',
                       choices=['comprehensive', 'paper', 'quick', 'comparison', 'hyperopt'])
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--physics', action='store_true', default=True)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--trials', type=int, default=10)
    
    args = parser.parse_args()
    
    exp = QuickExperiments()
    
    if args.mode == 'comprehensive':
        run_comprehensive_evaluation()
    
    elif args.mode == 'paper':
        run_paper_experiments()
    
    elif args.mode == 'quick':
        exp.quick_single_run(args.dataset, args.physics)
    
    elif args.mode == 'comparison':
        exp.compare_physics_vs_baseline([args.dataset], args.runs)
    
    elif args.mode == 'hyperopt':
        exp.hyperparameter_search(args.dataset, args.trials)