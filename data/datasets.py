"""
Unified dataset loaders for MNIST and CIFAR-10.

This module provides a consistent interface for loading and preprocessing
both MNIST and CIFAR-10 datasets for Vision Transformer experiments.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional, Dict, Any


def get_dataloaders(
    dataset: str,
    batch_size: int,
    data_dir: str = './data',
    num_workers: int = 2,
    pin_memory: bool = True,
    augmentation: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Get train and test dataloaders for specified dataset.

    Args:
        dataset: Dataset name ('mnist' or 'cifar10')
        batch_size: Batch size for both train and test loaders
        data_dir: Directory to download/load data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU transfer
        augmentation: Whether to apply data augmentation (training only)
        config: Optional config dict to use instead of loading from configs/

    Returns:
        Tuple of (train_loader, test_loader, config_dict)

    Raises:
        ValueError: If dataset name is not recognized
    """
    dataset_lower = dataset.lower()

    if dataset_lower == 'mnist':
        if config is None:
            from configs.mnist_config import MNIST_CONFIG as config

        # Training transforms
        if augmentation:
            train_transform = transforms.Compose([
                transforms.RandomRotation(10),  # Slight rotation
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])

        # Test transforms (no augmentation)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std'])
        ])

        # Load datasets
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform
        )

        print(f"Loaded MNIST: {len(train_dataset)} train, {len(test_dataset)} test samples")

    elif dataset_lower == 'cifar10':
        if config is None:
            from configs.cifar10_config import CIFAR10_CONFIG as config

        # Training transforms
        if augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])

        # Test transforms (no augmentation)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std'])
        ])

        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform
        )

        print(f"Loaded CIFAR-10: {len(train_dataset)} train, {len(test_dataset)} test samples")

    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. Supported datasets: 'mnist', 'cifar10'"
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for consistent batch sizes
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    # Print dataset info
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Add dataset info to config
    config['train_samples'] = len(train_dataset)
    config['test_samples'] = len(test_dataset)
    config['train_batches'] = len(train_loader)
    config['test_batches'] = len(test_loader)

    return train_loader, test_loader, config


def get_sample_batch(
    dataset: str,
    batch_size: int = 4,
    data_dir: str = './data'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a sample batch for testing model architecture.

    Args:
        dataset: Dataset name ('mnist' or 'cifar10')
        batch_size: Number of samples in batch
        data_dir: Directory to download/load data

    Returns:
        Tuple of (images, labels) tensors
    """
    train_loader, _, _ = get_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        data_dir=data_dir,
        num_workers=0,  # Use main thread for quick loading
        pin_memory=False
    )

    images, labels = next(iter(train_loader))
    return images, labels


def visualize_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    dataset: str,
    num_images: int = 8
):
    """
    Visualize a batch of images (requires matplotlib).

    Args:
        images: Batch of images (B, C, H, W)
        labels: Batch of labels (B,)
        dataset: Dataset name for proper class names
        num_images: Number of images to display
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Cannot visualize batch.")
        return

    # Class names
    if dataset.lower() == 'mnist':
        class_names = [str(i) for i in range(10)]
    else:  # CIFAR-10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']

    # Denormalize images for visualization
    if dataset.lower() == 'mnist':
        from configs.mnist_config import MNIST_CONFIG
        mean = MNIST_CONFIG['mean'][0]
        std = MNIST_CONFIG['std'][0]
        images = images * std + mean
    else:  # CIFAR-10
        from configs.cifar10_config import CIFAR10_CONFIG
        mean = torch.tensor(CIFAR10_CONFIG['mean']).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_CONFIG['std']).view(1, 3, 1, 1)
        images = images * std + mean

    # Clamp to valid range
    images = torch.clamp(images, 0, 1)

    # Create figure
    num_images = min(num_images, images.shape[0])
    fig, axes = plt.subplots(1, num_images, figsize=(2*num_images, 2))
    if num_images == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if dataset.lower() == 'mnist':
            # Grayscale image
            img = images[i].squeeze().cpu().numpy()
            ax.imshow(img, cmap='gray')
        else:
            # RGB image
            img = images[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)

        ax.set_title(class_names[labels[i].item()])
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def compute_dataset_stats(
    dataset: str,
    data_dir: str = './data',
    num_batches: Optional[int] = None
) -> Dict[str, Tuple[float, ...]]:
    """
    Compute mean and std for a dataset (for verification).

    Args:
        dataset: Dataset name ('mnist' or 'cifar10')
        data_dir: Directory to download/load data
        num_batches: Number of batches to use (None for all)

    Returns:
        Dictionary with 'mean' and 'std' tuples
    """
    # Create dataset without normalization
    transform = transforms.ToTensor()

    if dataset.lower() == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
    else:  # CIFAR-10
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )

    loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)

    # Compute mean and std
    mean = 0.0
    std = 0.0
    total_samples = 0

    for batch_idx, (data, _) in enumerate(loader):
        if num_batches and batch_idx >= num_batches:
            break

        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return {
        'mean': tuple(mean.numpy()),
        'std': tuple(std.numpy())
    }