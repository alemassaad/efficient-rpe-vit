"""
Multi-run benchmark script for Vision Transformer models.

This script orchestrates multiple training runs with different random seeds
to produce statistically rigorous performance metrics for model comparison.

Usage:
    # Benchmark specific models with 5 runs each
    python experiments/benchmark.py \
        --models baseline performer_favor performer_favor_most_general \
        --dataset mnist \
        --num-runs 5 \
        --epochs 20

    # Quick test with 2 runs
    python experiments/benchmark.py \
        --models baseline performer_favor \
        --dataset mnist \
        --num-runs 2 \
        --epochs 2 \
        --batch-size 256
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import list_available_models
from experiments.utils.benchmark_utils import (
    compute_aggregated_statistics,
    save_run_results,
    save_aggregated_statistics,
    save_benchmark_config,
    print_model_summary,
    load_run_metrics,
    format_time
)


def parse_benchmark_args():
    """Parse command-line arguments for benchmark."""
    parser = argparse.ArgumentParser(
        description='Run multi-seed benchmarks for Vision Transformer models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark 3 models with 5 runs each
  python experiments/benchmark.py --models baseline performer_favor performer_favor_most_general \\
      --dataset mnist --num-runs 5 --epochs 20

  # Quick test with 2 runs, 2 epochs
  python experiments/benchmark.py --models baseline performer_favor \\
      --dataset mnist --num-runs 2 --epochs 2 --batch-size 256
        """
    )

    # Model selection (required)
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help=f'Model variants to benchmark. Available: {", ".join(list_available_models())}'
    )

    # Dataset selection
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10'],
                       help='Dataset to use (default: mnist)')

    # Number of runs and seeds
    parser.add_argument('--num-runs', type=int, default=5,
                       help='Number of runs per model (default: 5)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                       help='Specific seeds to use (overrides --num-runs)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (uses config default if not specified)')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer type (default: adam)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['none', 'cosine', 'step'],
                       help='LR scheduler type (default: cosine)')

    # Output options
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory (auto-generated if not specified)')
    parser.add_argument('--save-checkpoints', action='store_true',
                       help='Save model checkpoints for each run')
    parser.add_argument('--plot-curves', action='store_true',
                       help='Generate training curve plots for each run')

    # Device options
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')

    # Resume options
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip runs that already have metrics.json (for resuming)')

    args = parser.parse_args()

    # Validate models
    available_models = list_available_models()
    for model in args.models:
        if model not in available_models:
            parser.error(f"Unknown model: {model}. Available: {', '.join(available_models)}")

    # Generate seeds if not specified
    if args.seeds is None:
        # Use reproducible default seeds
        args.seeds = [42 + i * 111 for i in range(args.num_runs)]
    else:
        # Update num_runs to match provided seeds
        args.num_runs = len(args.seeds)

    return args


def setup_benchmark_directory(args) -> Path:
    """
    Create directory structure for benchmark results.

    Args:
        args: Parsed command-line arguments

    Returns:
        Path to benchmark directory
    """
    if args.output_dir:
        benchmark_dir = Path(args.output_dir)
    else:
        # Auto-generate: results/benchmark_mnist_20250110_143022
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        benchmark_dir = Path('results') / f'benchmark_{args.dataset}_{timestamp}'

    benchmark_dir.mkdir(parents=True, exist_ok=True)

    return benchmark_dir


def run_single_training(
    model_name: str,
    dataset: str,
    seed: int,
    epochs: int,
    batch_size: int,
    output_dir: Path,
    config_overrides
) -> Dict:
    """
    Execute a single training run by calling train.py as subprocess.

    Args:
        model_name: Name of model to train
        dataset: Dataset to use
        seed: Random seed
        epochs: Number of epochs
        batch_size: Batch size
        output_dir: Directory for this run's outputs
        config_overrides: Additional configuration from args

    Returns:
        Metrics dict from the training run

    Raises:
        RuntimeError: If training fails
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        'experiments/train.py',
        '--model', model_name,
        '--dataset', dataset,
        '--seed', str(seed),
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--output-dir', str(output_dir),
        '--save-metrics',  # Always save metrics for benchmark
        '--optimizer', config_overrides.optimizer,
        '--scheduler', config_overrides.scheduler,
    ]

    # Add optional arguments
    if config_overrides.lr:
        cmd.extend(['--lr', str(config_overrides.lr)])
    if config_overrides.save_checkpoints:
        cmd.append('--save-model')
    if config_overrides.plot_curves:
        cmd.append('--save-plots')
    if config_overrides.cpu:
        cmd.append('--cpu')

    print(f"Running: {' '.join(cmd)}\n")

    # Run training
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed for {model_name} with seed {seed}")

    # Load metrics from saved file
    metrics_file = output_dir / f"{model_name}_{dataset}_metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    return metrics


def main():
    """Main benchmark orchestration."""
    args = parse_benchmark_args()

    print(f"\n{'='*80}")
    print("MULTI-RUN BENCHMARK")
    print(f"{'='*80}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Runs per model: {args.num_runs}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*80}\n")

    # Setup output directory
    benchmark_dir = setup_benchmark_directory(args)
    print(f"Results will be saved to: {benchmark_dir}\n")

    # Save benchmark configuration
    save_benchmark_config(benchmark_dir, args)

    # Track overall progress
    total_runs = len(args.models) * args.num_runs
    completed_runs = 0
    overall_start_time = time.time()

    # Loop over models
    for model_idx, model_name in enumerate(args.models):
        print(f"\n{'#'*80}")
        print(f"# MODEL {model_idx + 1}/{len(args.models)}: {model_name}")
        print(f"{'#'*80}\n")

        model_dir = benchmark_dir / model_name
        model_dir.mkdir(exist_ok=True)

        # Run multiple times with different seeds
        run_results = []
        for run_idx, seed in enumerate(args.seeds):
            run_output_dir = model_dir / f"run_{run_idx}_seed_{seed}"

            # Check if run already exists (for resuming)
            if args.skip_existing:
                existing_metrics = load_run_metrics(run_output_dir)
                if existing_metrics is not None:
                    print(f"{'='*80}")
                    print(f"Skipping existing run: {model_name} | Run {run_idx+1}/{len(args.seeds)} | Seed {seed}")
                    print(f"{'='*80}\n")
                    run_results.append(existing_metrics)
                    completed_runs += 1
                    continue

            print(f"{'='*80}")
            print(f"Model: {model_name} | Run: {run_idx+1}/{len(args.seeds)} | Seed: {seed}")
            print(f"Overall progress: {completed_runs}/{total_runs} runs completed")
            print(f"{'='*80}\n")

            run_start_time = time.time()

            try:
                # Execute single training run
                run_metrics = run_single_training(
                    model_name=model_name,
                    dataset=args.dataset,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    output_dir=run_output_dir,
                    config_overrides=args
                )

                # Save individual run results (already saved by train.py, but we save again for safety)
                # save_run_results(run_metrics, run_output_dir)

                run_results.append(run_metrics)

                run_time = time.time() - run_start_time
                completed_runs += 1

                print(f"\n{'='*80}")
                print(f"Run completed in {format_time(run_time)}")
                print(f"Test Accuracy: {run_metrics['aggregate']['best_test_accuracy']:.2f}%")
                print(f"{'='*80}\n")

            except Exception as e:
                print(f"\n{'!'*80}")
                print(f"ERROR: Run failed with exception: {e}")
                print(f"{'!'*80}\n")
                # Continue with next run rather than stopping entire benchmark
                continue

        # Check if we have any successful runs
        if not run_results:
            print(f"\n{'!'*80}")
            print(f"WARNING: No successful runs for {model_name}, skipping aggregation")
            print(f"{'!'*80}\n")
            continue

        # Compute and save aggregated statistics
        print(f"\n{'='*80}")
        print(f"Computing aggregated statistics for {model_name}...")
        print(f"{'='*80}\n")

        aggregated = compute_aggregated_statistics(run_results)
        save_aggregated_statistics(aggregated, model_dir)

        # Print summary for this model
        print_model_summary(model_name, aggregated)

    # Final summary
    total_time = time.time() - overall_start_time

    print(f"\n{'#'*80}")
    print("# BENCHMARK COMPLETE")
    print(f"{'#'*80}")
    print(f"Total runs: {completed_runs}/{total_runs}")
    print(f"Total time: {format_time(total_time)}")
    print(f"Results saved to: {benchmark_dir}")
    print(f"{'#'*80}\n")

    # Print summary table of all models
    print("\nSummary Table:")
    print(f"{'='*80}")
    print(f"{'Model':<35} {'Runs':<6} {'Best Acc (%)':<20} {'Train Time (s)':<15}")
    print(f"{'='*80}")

    for model_name in args.models:
        stats_file = benchmark_dir / model_name / 'aggregated_stats.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                agg = json.load(f)
            stats = agg['statistics']

            if 'best_test_accuracy' in stats and stats['best_test_accuracy']['mean'] is not None:
                acc_mean = stats['best_test_accuracy']['mean']
                acc_std = stats['best_test_accuracy']['std']
                acc_str = f"{acc_mean:.2f} ± {acc_std:.2f}"
            else:
                acc_str = "N/A"

            if 'avg_train_time_per_epoch' in stats and stats['avg_train_time_per_epoch']['mean'] is not None:
                time_mean = stats['avg_train_time_per_epoch']['mean']
                time_std = stats['avg_train_time_per_epoch']['std']
                time_str = f"{time_mean:.1f} ± {time_std:.1f}"
            else:
                time_str = "N/A"

            print(f"{model_name:<35} {agg['num_runs']:<6} {acc_str:<20} {time_str:<15}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
