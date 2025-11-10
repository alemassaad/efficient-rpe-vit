"""
Common training utilities for all Vision Transformer models.

This module contains shared functions for training, evaluation, and benchmarking
that work with any model variant.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import numpy as np

# Import native PyTorch metrics (no sklearn overhead!)
from .metrics import compute_classification_metrics


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int = 100,
    global_start_time: Optional[float] = None
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
        Dictionary with epoch metrics including loss, accuracy, and time
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

        # Log progress every 2%
        current_pct = int((batch_idx / len(train_loader)) * 100)
        prev_pct = int(((batch_idx - 1) / len(train_loader)) * 100) if batch_idx > 0 else -1

        # Print when percentage crosses a 2% boundary
        if batch_idx == 0 or (current_pct // 2) > (prev_pct // 2):
            elapsed = time.time() - global_start_time if global_start_time else 0
            print(f'[{elapsed:6.1f}s] Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')

    # Print final 100% progress
    elapsed = time.time() - global_start_time if global_start_time else 0
    print(f'[{elapsed:6.1f}s] Train Epoch: {epoch} [{len(train_loader)}/{len(train_loader)} '
          f'(100%)\t'
          f'Loss: {loss.item():.6f}')

    # Calculate metrics
    epoch_time = time.time() - start_time
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total

    # Get peak memory if using GPU
    peak_memory = 0
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # Convert to MB
        torch.cuda.reset_peak_memory_stats(device)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'time': epoch_time,
        'peak_memory_mb': peak_memory
    }


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_predictions: bool = False,
    log_interval: int = 100,
    global_start_time: Optional[float] = None,
    epoch: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate model on test/validation set.

    Args:
        model: Model to evaluate
        test_loader: Test/validation data loader
        criterion: Loss function
        device: Device to use
        return_predictions: If True, return predictions and labels

    Returns:
        Dictionary with test metrics including loss, accuracy, and optionally
        precision, recall, F1-score
    """
    model.eval()
    test_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            # Log progress every 2%
            current_pct = int((batch_idx / len(test_loader)) * 100)
            prev_pct = int(((batch_idx - 1) / len(test_loader)) * 100) if batch_idx > 0 else -1

            # Print when percentage crosses a 2% boundary
            if batch_idx == 0 or (current_pct // 2) > (prev_pct // 2):
                elapsed = time.time() - global_start_time if global_start_time else 0
                epoch_str = f" Epoch: {epoch}" if epoch is not None else ""
                print(f'[{elapsed:6.1f}s] Test{epoch_str} [{batch_idx}/{len(test_loader)} '
                      f'({100. * batch_idx / len(test_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.6f}')

            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                all_predictions.append(predicted)
                all_labels.append(labels)

    # Print final 100% progress
    elapsed = time.time() - global_start_time if global_start_time else 0
    epoch_str = f" Epoch: {epoch}" if epoch is not None else ""
    print(f'[{elapsed:6.1f}s] Test{epoch_str} [{len(test_loader)}/{len(test_loader)} '
          f'(100%)\t'
          f'Loss: {loss.item():.6f}')

    # Calculate basic metrics - always use tensors for consistency
    if not return_predictions:
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
    else:
        # Convert numpy arrays back to tensors
        all_predictions = torch.tensor(all_predictions)
        all_labels = torch.tensor(all_labels)

    correct = all_predictions.eq(all_labels).sum().item()
    total = all_labels.size(0)

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total

    results = {
        'loss': avg_loss,
        'accuracy': accuracy
    }

    # Calculate additional metrics if predictions were collected
    if return_predictions:
        # Use native PyTorch implementation (no sklearn import overhead!)
        num_classes = max(all_labels.max().item(), all_predictions.max().item()) + 1
        precision, recall, f1 = compute_classification_metrics(
            all_predictions, all_labels, num_classes, average='weighted'
        )

        results.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_predictions.numpy(),
            'labels': all_labels.numpy()
        })

    return results


def benchmark_inference(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_warmup: int = 10,
    num_benchmark: int = 100
) -> Dict[str, float]:
    """
    Benchmark inference performance.

    Args:
        model: Model to benchmark
        test_loader: Test data loader
        device: Device to use
        num_warmup: Number of warmup iterations
        num_benchmark: Number of benchmark iterations

    Returns:
        Dictionary with inference metrics including throughput, latency, and memory
    """
    model.eval()

    # Get a batch for benchmarking
    images, _ = next(iter(test_loader))
    batch_size = images.size(0)
    images = images.to(device)

    # Warmup
    print(f"Running {num_warmup} warmup iterations...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(images)

    # Synchronize if using GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    print(f"Running {num_benchmark} benchmark iterations...")
    latencies = []

    with torch.no_grad():
        for _ in range(num_benchmark):
            start = time.time()

            _ = model(images)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            latencies.append(time.time() - start)

    # Calculate statistics
    latencies = np.array(latencies)
    total_time = latencies.sum()
    total_images = batch_size * num_benchmark

    # Get memory usage if using GPU
    peak_memory = 0
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

    return {
        'total_time_s': total_time,
        'total_images': total_images,
        'throughput_imgs_per_s': total_images / total_time,
        'avg_latency_ms': latencies.mean() * 1000,
        'std_latency_ms': latencies.std() * 1000,
        'min_latency_ms': latencies.min() * 1000,
        'max_latency_ms': latencies.max() * 1000,
        'batch_size': batch_size,
        'num_iterations': num_benchmark,
        'peak_memory_mb': peak_memory
    }


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.0,
    optimizer_type: str = 'adam'
) -> optim.Optimizer:
    """
    Create optimizer for training.

    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')

    Returns:
        Configured optimizer
    """
    if optimizer_type.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_lr_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'cosine',
    num_epochs: int = 100,
    warmup_epochs: int = 0
) -> Optional[object]:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ('cosine', 'step', 'none')
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs

    Returns:
        Configured scheduler or None
    """
    if scheduler_type == 'none':
        return None

    if scheduler_type == 'cosine':
        if warmup_epochs > 0:
            # Use linear warmup + cosine annealing
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))

            return LambdaLR(optimizer, lr_lambda)
        else:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, T_max=num_epochs)

    elif scheduler_type == 'step':
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=30, gamma=0.1)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict,
    filepath: str,
    model_name: str = None
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        metrics: Current metrics
        filepath: Path to save checkpoint
        model_name: Optional model name for metadata
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    # Add model metadata if available
    if hasattr(model, 'model_name'):
        checkpoint['model_name'] = model.model_name
    elif model_name:
        checkpoint['model_name'] = model_name

    if hasattr(model, 'attention_type'):
        checkpoint['attention_type'] = model.attention_type

    if hasattr(model, 'rpe_type'):
        checkpoint['rpe_type'] = model.rpe_type

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    filepath: str
) -> Tuple[int, Dict]:
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        filepath: Path to checkpoint

    Returns:
        Tuple of (epoch, metrics) from checkpoint
    """
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})

    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")

    return epoch, metrics