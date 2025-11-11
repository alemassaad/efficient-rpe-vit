"""
Benchmark-specific utility functions for multi-run experiments.

This module provides helper functions for:
- Computing convergence metrics
- Aggregating statistics across runs
- Saving and loading benchmark results
- Setting random seeds for reproducibility
"""

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


def compute_convergence_metrics(per_epoch_data: List[Dict]) -> Dict:
    """
    Compute convergence metrics from per-epoch training history.

    Args:
        per_epoch_data: List of dicts with 'epoch', 'test_accuracy', etc.

    Returns:
        Dict with convergence metrics:
        - epochs_to_90_percent: First epoch reaching ≥90% test accuracy
        - epochs_to_95_percent: First epoch reaching ≥95% test accuracy
        - epochs_to_99_percent: First epoch reaching ≥99% test accuracy
        - epochs_until_plateau: Epoch when test accuracy stops improving
    """
    convergence = {
        'epochs_to_90_percent': None,
        'epochs_to_95_percent': None,
        'epochs_to_99_percent': None,
        'epochs_until_plateau': None,
    }

    if not per_epoch_data:
        return convergence

    # Find first epoch reaching each threshold
    for epoch_data in per_epoch_data:
        acc = epoch_data['test_accuracy']
        epoch = epoch_data['epoch']

        if convergence['epochs_to_90_percent'] is None and acc >= 90.0:
            convergence['epochs_to_90_percent'] = epoch
        if convergence['epochs_to_95_percent'] is None and acc >= 95.0:
            convergence['epochs_to_95_percent'] = epoch
        if convergence['epochs_to_99_percent'] is None and acc >= 99.0:
            convergence['epochs_to_99_percent'] = epoch

    # Detect plateau (no improvement >0.1% for 3 consecutive epochs)
    plateau_threshold = 0.1  # percent
    window_size = 3

    if len(per_epoch_data) >= window_size:
        for i in range(len(per_epoch_data) - window_size + 1):
            window = per_epoch_data[i:i+window_size]
            accuracies = [e['test_accuracy'] for e in window]

            # Check if all accuracies in window are within threshold
            if max(accuracies) - min(accuracies) <= plateau_threshold:
                convergence['epochs_until_plateau'] = window[0]['epoch']
                break

    return convergence


def compute_aggregated_statistics(run_results: List[Dict]) -> Dict:
    """
    Compute mean, std, min, max across multiple runs.

    Args:
        run_results: List of metrics dicts from individual runs

    Returns:
        Dict with aggregated statistics for each metric
    """
    if not run_results:
        raise ValueError("run_results cannot be empty")

    # Determine which metrics to aggregate
    aggregate_keys = run_results[0]['aggregate'].keys()
    inference_keys = run_results[0]['inference'].keys()

    aggregated = {
        'model': run_results[0]['metadata']['model'],
        'dataset': run_results[0]['metadata']['dataset'],
        'num_runs': len(run_results),
        'seeds': [r['metadata']['seed'] for r in run_results],
        'statistics': {}
    }

    # Aggregate each metric from 'aggregate' section
    for key in aggregate_keys:
        values = [run['aggregate'][key] for run in run_results]

        # Handle None values (e.g., epochs_to_99_percent might be None)
        values_clean = [v for v in values if v is not None]

        if len(values_clean) > 0:
            aggregated['statistics'][key] = {
                'mean': float(np.mean(values_clean)),
                'std': float(np.std(values_clean, ddof=1)) if len(values_clean) > 1 else 0.0,
                'min': float(np.min(values_clean)),
                'max': float(np.max(values_clean)),
                'values': values  # Keep original including None
            }
        else:
            # All values were None
            aggregated['statistics'][key] = {
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'values': values
            }

    # Aggregate inference metrics
    for key in inference_keys:
        values = [run['inference'][key] for run in run_results]
        values_clean = [v for v in values if v is not None]

        if len(values_clean) > 0:
            aggregated['statistics'][key] = {
                'mean': float(np.mean(values_clean)),
                'std': float(np.std(values_clean, ddof=1)) if len(values_clean) > 1 else 0.0,
                'min': float(np.min(values_clean)),
                'max': float(np.max(values_clean)),
                'values': values
            }

    return aggregated


def save_run_results(metrics: Dict, output_dir: Path):
    """
    Save individual run results to JSON file.

    Args:
        metrics: Complete metrics dict from single run
        output_dir: Directory for this run
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"Run results saved to {metrics_file}")


def save_aggregated_statistics(aggregated: Dict, model_dir: Path):
    """
    Save aggregated statistics across runs.

    Args:
        aggregated: Statistics dict from compute_aggregated_statistics()
        model_dir: Directory for this model
    """
    stats_file = model_dir / 'aggregated_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)

    print(f"Aggregated statistics saved to {stats_file}")


def save_benchmark_config(benchmark_dir: Path, args):
    """
    Save benchmark configuration for reproducibility.

    Args:
        benchmark_dir: Root benchmark directory
        args: Parsed command-line arguments
    """
    config = {
        'models': args.models,
        'dataset': args.dataset,
        'num_runs': len(args.seeds),
        'seeds': args.seeds,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'command': ' '.join(sys.argv)
    }

    config_file = benchmark_dir / 'benchmark_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Benchmark configuration saved to {config_file}")


def print_model_summary(model_name: str, aggregated: Dict):
    """
    Print human-readable summary of model performance.

    Args:
        model_name: Name of the model
        aggregated: Aggregated statistics dict
    """
    stats = aggregated['statistics']

    print(f"\n{'='*80}")
    print(f"Summary: {model_name}")
    print(f"{'='*80}")
    print(f"Runs: {aggregated['num_runs']}")
    print(f"Seeds: {aggregated['seeds']}")
    print()

    # Best test accuracy
    if 'best_test_accuracy' in stats and stats['best_test_accuracy']['mean'] is not None:
        s = stats['best_test_accuracy']
        print(f"Best Test Accuracy:  {s['mean']:.2f}% ± {s['std']:.2f}%")
        print(f"  Range: [{s['min']:.2f}%, {s['max']:.2f}%]")

    # Final test accuracy
    if 'final_test_accuracy' in stats and stats['final_test_accuracy']['mean'] is not None:
        s = stats['final_test_accuracy']
        print(f"Final Test Accuracy: {s['mean']:.2f}% ± {s['std']:.2f}%")

    # Training time
    if 'avg_train_time_per_epoch' in stats and stats['avg_train_time_per_epoch']['mean'] is not None:
        s = stats['avg_train_time_per_epoch']
        print(f"Train Time/Epoch:    {s['mean']:.2f}s ± {s['std']:.2f}s")

    # Total training time
    if 'total_training_time' in stats and stats['total_training_time']['mean'] is not None:
        s = stats['total_training_time']
        print(f"Total Training Time: {s['mean']:.2f}s ± {s['std']:.2f}s")

    # Inference latency
    if 'inference_latency_mean_ms' in stats and stats['inference_latency_mean_ms']['mean'] is not None:
        s = stats['inference_latency_mean_ms']
        print(f"Inference Latency:   {s['mean']:.2f}ms ± {s['std']:.2f}ms")

    # Convergence to 95%
    if 'epochs_to_95_percent' in stats and stats['epochs_to_95_percent']['mean'] is not None:
        s = stats['epochs_to_95_percent']
        print(f"Epochs to 95%:       {s['mean']:.1f} ± {s['std']:.1f}")

    # Plateau detection
    if 'epochs_until_plateau' in stats and stats['epochs_until_plateau']['mean'] is not None:
        s = stats['epochs_until_plateau']
        print(f"Epochs until Plateau: {s['mean']:.1f} ± {s['std']:.1f}")

    print(f"{'='*80}\n")


def set_random_seeds(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # For reproducibility (may impact performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_run_metrics(run_dir: Path) -> Optional[Dict]:
    """
    Load metrics from a completed run.

    Args:
        run_dir: Directory containing metrics.json

    Returns:
        Metrics dict or None if file doesn't exist
    """
    metrics_file = run_dir / 'metrics.json'
    if not metrics_file.exists():
        return None

    with open(metrics_file, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string like "2h 34m" or "45.3s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
