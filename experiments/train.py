"""
Unified training script for all Vision Transformer variants.

This script can train any model variant (baseline, performer_favor, etc.)
on either MNIST or CIFAR-10, with comprehensive metrics tracking.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_model, list_available_models, get_model_info
from data.datasets import get_dataloaders, visualize_batch, get_sample_batch
from experiments.utils import (
    train_epoch,
    evaluate,
    benchmark_inference,
    create_optimizer,
    create_lr_scheduler,
    save_checkpoint,
    load_checkpoint
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Vision Transformer models with various attention mechanisms and RPE types'
    )

    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help=f'Model variant to train. Available: {", ".join(list_available_models())}'
    )

    # Dataset selection
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        choices=['mnist', 'cifar10'],
        help='Dataset to use (default: mnist)'
    )

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Weight decay (overrides config)')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate (overrides config)')

    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type (default: adam)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'step'],
                        help='LR scheduler type (default: cosine)')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='Number of warmup epochs')

    # Data options
    parser.add_argument('--augmentation', action='store_true',
                        help='Enable data augmentation')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loader workers')

    # Device options
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Output options
    parser.add_argument('--save-model', action='store_true',
                        help='Save model checkpoint')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Save training metrics')
    parser.add_argument('--plot', action='store_true',
                        help='Display training curves')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save training curve plots')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize sample batch before training')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-generated)')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Logging
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Log every N batches')

    return parser.parse_args()


def load_config(dataset: str) -> Dict[str, Any]:
    """Load dataset configuration."""
    if dataset == 'mnist':
        from configs.mnist_config import MNIST_CONFIG
        return MNIST_CONFIG.copy()
    elif dataset == 'cifar10':
        from configs.cifar10_config import CIFAR10_CONFIG
        return CIFAR10_CONFIG.copy()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main():
    """Main training function."""
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device setup
    if args.cpu:
        device = torch.device('cpu')
        print("Using CPU")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU")

    # Load configuration
    config = load_config(args.dataset)

    # Override config with command-line arguments
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.lr is not None:
        config['learning_rate'] = args.lr
    if args.weight_decay is not None:
        config['weight_decay'] = args.weight_decay
    if args.dropout is not None:
        config['dropout'] = args.dropout

    # Get model info
    model_info = get_model_info(args.model)
    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Attention: {model_info['attention_type']} ({model_info['attention_complexity']})")
    print(f"RPE: {model_info['rpe_type'] or 'None'}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"{'='*60}\n")

    # Create data loaders
    train_loader, test_loader, data_config = get_dataloaders(
        dataset=args.dataset,
        batch_size=config['batch_size'],
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        augmentation=args.augmentation,
        config=config
    )

    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Training batches: {len(train_loader)}")

    # Visualize sample batch if requested
    if args.visualize:
        sample_batch = get_sample_batch(train_loader, device)
        if sample_batch:
            visualize_batch(*sample_batch, num_samples=8)

    # Create model
    print(f"\nCreating {args.model} model...")
    model = create_model(args.model, config)
    model = model.to(device)

    # Count parameters
    param_counts = model.count_parameters()
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        config['learning_rate'],
        config.get('weight_decay', 0.0),
        args.optimizer
    )

    scheduler = create_lr_scheduler(
        optimizer,
        args.scheduler,
        config['epochs'],
        args.warmup_epochs
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1  # Start from next epoch

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"results/{args.model}_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}")

    if args.save_model or args.save_metrics or args.save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {config['epochs']} epochs...")
    print(f"{'='*60}\n")

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    best_test_acc = 0

    global_start_time = time.time()

    for epoch in range(start_epoch, config['epochs'] + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, args.log_interval, global_start_time
        )
        train_losses.append(train_metrics['loss'])
        train_accs.append(train_metrics['accuracy'])

        # Evaluate
        test_metrics = evaluate(
            model, test_loader, criterion, device,
            log_interval=args.log_interval,
            global_start_time=global_start_time,
            epoch=epoch
        )
        test_losses.append(test_metrics['loss'])
        test_accs.append(test_metrics['accuracy'])

        # Get current learning rate before updating
        if scheduler:
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = config['learning_rate']

        # Update learning rate for next epoch
        if scheduler:
            scheduler.step()

        # Print epoch summary
        print(f"\nEpoch {epoch}/{config['epochs']} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Test Loss: {test_metrics['loss']:.4f}, "
              f"Test Acc: {test_metrics['accuracy']:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Epoch Time: {train_metrics['time']:.2f}s")
        if train_metrics.get('peak_memory_mb', 0) > 0:
            print(f"  Peak Memory: {train_metrics['peak_memory_mb']:.2f} MB")
        print("-" * 60)

        # Track best model accuracy
        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']

            # Save checkpoint if requested
            if args.save_model:
                checkpoint_path = output_dir / f"{args.model}_{args.dataset}_best.pth"
                save_checkpoint(
                    model, optimizer, epoch, test_metrics,
                    str(checkpoint_path), args.model
                )

    # Final evaluation
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Total training time: {time.time() - global_start_time:.2f}s")
    print(f"Best test accuracy: {best_test_acc:.2f}%")

    # Benchmark inference
    print(f"\n{'='*60}")
    print("Benchmarking inference performance...")
    benchmark_metrics = benchmark_inference(model, test_loader, device)

    print(f"Throughput: {benchmark_metrics['throughput_imgs_per_s']:.2f} images/second")
    print(f"Average latency: {benchmark_metrics['avg_latency_ms']:.2f}ms")
    print(f"Latency std: {benchmark_metrics['std_latency_ms']:.2f}ms")
    if benchmark_metrics.get('peak_memory_mb', 0) > 0:
        print(f"Peak memory: {benchmark_metrics['peak_memory_mb']:.2f} MB")

    # Save metrics
    if args.save_metrics:
        metrics = {
            'model': args.model,
            'attention_type': model_info['attention_type'],
            'rpe_type': model_info['rpe_type'],
            'dataset': args.dataset,
            'config': config,
            'parameters': param_counts,
            'training': {
                'train_losses': train_losses,
                'train_accuracies': train_accs,
                'test_losses': test_losses,
                'test_accuracies': test_accs,
                'best_test_accuracy': best_test_acc,
                'total_time': time.time() - global_start_time
            },
            'inference': benchmark_metrics,
            'args': vars(args)
        }

        metrics_path = output_dir / f"{args.model}_{args.dataset}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Metrics saved to {metrics_path}")

    # Plot training curves
    if args.plot or args.save_plots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        epochs_range = range(start_epoch, start_epoch + len(train_losses))

        # Loss plot
        ax1.plot(epochs_range, train_losses, label='Train Loss')
        ax1.plot(epochs_range, test_losses, label='Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{args.model} - Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs_range, train_accs, label='Train Acc')
        ax2.plot(epochs_range, test_accs, label='Test Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(f'{args.model} - Accuracy Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'{args.model} on {args.dataset.upper()}')
        plt.tight_layout()

        if args.save_plots:
            plot_path = output_dir / f"{args.model}_{args.dataset}_curves.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to {plot_path}")

        if args.plot:
            plt.show()

        plt.close()

    print(f"\n{'='*60}")
    print("Training script completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()