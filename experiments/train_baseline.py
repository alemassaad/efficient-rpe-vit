"""
Training script for Baseline Vision Transformer.

This script trains the BaselineViT model on either MNIST or CIFAR-10 dataset,
tracking comprehensive metrics for comparison with Performer variants.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline_vit import BaselineViT, create_baseline_vit
from data.datasets import get_dataloaders, visualize_batch, get_sample_batch


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int = 100,
    global_start_time: float = None
) -> Dict[str, float]:
    """
    Train model for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        epoch: Current epoch number
        log_interval: Log every N batches
        global_start_time: Global training start time for elapsed time display

    Returns:
        Dictionary with epoch metrics
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Log progress
        if batch_idx % log_interval == 0:
            elapsed = time.time() - global_start_time if global_start_time else 0
            print(f'[{elapsed:6.1f}s] Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')

    # Calculate metrics
    epoch_time = time.time() - start_time
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'time': epoch_time
    }


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to use

    Returns:
        Dictionary with test metrics
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def benchmark_inference(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    global_start_time: float = None
) -> Dict[str, float]:
    """
    Benchmark inference performance.

    Args:
        model: Model to benchmark
        test_loader: Test data loader
        device: Device to use
        global_start_time: Global training start time for elapsed time display

    Returns:
        Dictionary with inference metrics
    """
    model.eval()

    # Warmup (important for accurate timing)
    if global_start_time:
        elapsed = time.time() - global_start_time
        print(f"[{elapsed:6.1f}s] Running warmup...")
    else:
        print("Running warmup...")

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            _ = model(images)
            break

    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Timed inference
    if global_start_time:
        elapsed = time.time() - global_start_time
        print(f"[{elapsed:6.1f}s] Running timed inference...")
    else:
        print("Running timed inference...")

    start_time = time.time()
    total_images = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            _ = model(images)
            total_images += images.size(0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    inference_time = time.time() - start_time
    throughput = total_images / inference_time

    # Get memory stats
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_memory_mb = 0

    return {
        'total_time': inference_time,
        'total_images': total_images,
        'throughput': throughput,
        'ms_per_image': (inference_time / total_images) * 1000,
        'peak_memory_mb': peak_memory_mb
    }


def main(args):
    """Main training function."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    # Load configuration
    if args.dataset.lower() == 'mnist':
        from configs.mnist_config import MNIST_CONFIG as config
    elif args.dataset.lower() == 'cifar10':
        from configs.cifar10_config import CIFAR10_CONFIG as config
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Override config with command-line arguments if provided
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr

    print(f"\nConfiguration for {args.dataset.upper()}:")
    print(json.dumps({k: v for k, v in config.items() if k != 'alternative_patch_sizes'},
                     indent=2))

    # Create data loaders
    print("\nLoading dataset...")
    train_loader, test_loader, config = get_dataloaders(
        dataset=args.dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'] and torch.cuda.is_available(),
        augmentation=args.augmentation or config.get('augmentation', False)
    )

    # Optionally visualize a batch
    if args.visualize:
        print("\nVisualizing sample batch...")
        images, labels = next(iter(train_loader))
        visualize_batch(images[:8], labels[:8], args.dataset)

    # Create model
    print("\nCreating model...")
    model = create_baseline_vit(config).to(device)

    # Print model info
    param_counts = model.count_parameters()
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.0)
    )

    # Learning rate scheduler (optional)
    if config.get('warmup_epochs', 0) > 0:
        # Simple linear warmup
        def warmup_lambda(epoch):
            if epoch < config['warmup_epochs']:
                return (epoch + 1) / config['warmup_epochs']
            return 1.0
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    else:
        scheduler = None

    # Storage for metrics
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_time': [],
        'peak_memory_mb': []
    }

    # Training loop
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)

    total_training_start = time.time()
    best_test_acc = 0

    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()

        # Reset GPU memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, config['log_interval'], total_training_start
        )

        # Evaluate
        if epoch % config.get('eval_frequency', 1) == 0:
            test_metrics = evaluate(model, test_loader, criterion, device)
        else:
            test_metrics = {'loss': 0, 'accuracy': 0}

        # Update learning rate
        if scheduler:
            scheduler.step()

        # Get memory stats
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        else:
            peak_memory = 0

        # Store metrics
        metrics['train_loss'].append(train_metrics['loss'])
        metrics['train_acc'].append(train_metrics['accuracy'])
        metrics['test_loss'].append(test_metrics['loss'])
        metrics['test_acc'].append(test_metrics['accuracy'])
        metrics['epoch_time'].append(time.time() - epoch_start)
        metrics['peak_memory_mb'].append(peak_memory)

        # Print epoch summary with elapsed time
        elapsed = time.time() - total_training_start
        print(f'\n[{elapsed:6.1f}s] Epoch {epoch}/{config["epochs"]}:')
        print(f'[{elapsed:6.1f}s]   Train Loss: {train_metrics["loss"]:.4f} | Train Acc: {train_metrics["accuracy"]:.2f}%')
        print(f'[{elapsed:6.1f}s]   Test Loss: {test_metrics["loss"]:.4f} | Test Acc: {test_metrics["accuracy"]:.2f}%')
        print(f'[{elapsed:6.1f}s]   Time: {metrics["epoch_time"][-1]:.2f}s | Peak Memory: {peak_memory:.1f} MB')

        # Save best model
        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']
            if args.save_model:
                checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f'baseline_vit_{args.dataset}_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_metrics['accuracy'],
                    'config': config
                }, checkpoint_path)
                print(f'[{elapsed:6.1f}s]   Saved best model to {checkpoint_path}')

        # Add blank line after each epoch for better readability
        print()

    total_training_time = time.time() - total_training_start

    # Inference benchmarking
    print("\n" + "="*50)
    print("INFERENCE BENCHMARKING")
    print("="*50)

    inference_metrics = benchmark_inference(model, test_loader, device, total_training_start)

    # Final results summary
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Model: BaselineViT")
    print(f"Parameters: {param_counts['total']:,}")

    print(f"\nTraining:")
    print(f"  Total time: {total_training_time:.2f}s")
    print(f"  Avg time per epoch: {sum(metrics['epoch_time'])/len(metrics['epoch_time']):.2f}s")
    print(f"  Final train accuracy: {metrics['train_acc'][-1]:.2f}%")
    print(f"  Final test accuracy: {metrics['test_acc'][-1]:.2f}%")
    print(f"  Best test accuracy: {best_test_acc:.2f}%")
    print(f"  Peak memory (training): {max(metrics['peak_memory_mb']):.1f} MB")

    print(f"\nInference:")
    print(f"  Total time: {inference_metrics['total_time']:.2f}s")
    print(f"  Throughput: {inference_metrics['throughput']:.1f} images/sec")
    print(f"  Avg time per image: {inference_metrics['ms_per_image']:.2f} ms")
    print(f"  Peak memory (inference): {inference_metrics['peak_memory_mb']:.1f} MB")

    # Check against expected performance
    if 'expected_accuracy' in config:
        expected_acc = config['expected_accuracy'] * 100
        actual_acc = metrics['test_acc'][-1]
        threshold = expected_acc * 0.95  # 95% of expected
        if actual_acc >= threshold:
            print(f"\n✓ Performance check PASSED:")
            print(f"  Actual accuracy: {actual_acc:.1f}%")
            print(f"  Minimum required: {threshold:.1f}% (95% of expected {expected_acc:.1f}%)")
        else:
            print(f"\n✗ Performance check FAILED:")
            print(f"  Actual accuracy: {actual_acc:.1f}%")
            print(f"  Minimum required: {threshold:.1f}% (95% of expected {expected_acc:.1f}%)")
            print(f"  Gap to threshold: {threshold - actual_acc:.1f}%")

    # Save metrics to file
    if args.save_metrics:
        metrics_dir = Path('./results')
        metrics_dir.mkdir(exist_ok=True)
        metrics_file = metrics_dir / f'baseline_vit_{args.dataset}_metrics.json'

        with open(metrics_file, 'w') as f:
            json.dump({
                'config': config,
                'training_metrics': metrics,
                'inference_metrics': inference_metrics,
                'best_test_accuracy': best_test_acc,
                'total_training_time': total_training_time,
                'parameter_count': param_counts
            }, f, indent=2)

        print(f"\nSaved metrics to {metrics_file}")

    # Plot training curves if requested
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # Loss curve
            axes[0].plot(metrics['train_loss'], label='Train')
            axes[0].plot(metrics['test_loss'], label='Test')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].legend()
            axes[0].grid(True)

            # Accuracy curves
            axes[1].plot(metrics['train_acc'], label='Train')
            axes[1].plot(metrics['test_acc'], label='Test')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title('Accuracy Over Time')
            axes[1].legend()
            axes[1].grid(True)

            # Training time per epoch
            axes[2].plot(metrics['epoch_time'])
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Time (s)')
            axes[2].set_title('Time per Epoch')
            axes[2].grid(True)

            plt.suptitle(f'BaselineViT Training on {args.dataset.upper()}')
            plt.tight_layout()

            if args.save_plots:
                plots_dir = Path('./plots')
                plots_dir.mkdir(exist_ok=True)
                plot_file = plots_dir / f'baseline_vit_{args.dataset}_curves.png'
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                print(f"Saved plot to {plot_file}")

            plt.show()
        except ImportError:
            print("matplotlib not installed. Skipping plots.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Baseline Vision Transformer')

    # Dataset and model
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10'],
                       help='Dataset to use (default: mnist)')

    # Training hyperparameters (override config)
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--augmentation', action='store_true',
                       help='Enable data augmentation')

    # System
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    # Output options
    parser.add_argument('--save-model', action='store_true',
                       help='Save best model checkpoint')
    parser.add_argument('--save-metrics', action='store_true',
                       help='Save metrics to JSON file')
    parser.add_argument('--plot', action='store_true',
                       help='Plot training curves')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to file')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize sample batch before training')

    args = parser.parse_args()
    main(args)