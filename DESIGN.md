# OOP Framework Design for Model Evaluation

**STATUS: TENTATIVE DESIGN - NOT YET IMPLEMENTED**

This document outlines our planned object-oriented framework for systematically testing and comparing different Vision Transformer architectures. This design is subject to change as we implement and refine the system.

---

## 1. Framework Overview

### Core Concept
Create a **model-agnostic evaluation framework** that can train, test, and benchmark any Vision Transformer architecture while collecting a comprehensive set of metrics. The evaluator handles all measurement concerns, allowing model implementations to focus solely on architecture.

### Design Philosophy
- **Separation of Concerns**: Models define architecture; Evaluator handles training, testing, and metrics
- **Consistency**: All 9 model variants evaluated with identical methodology
- **Reproducibility**: Same datasets, same hyperparameters, same measurement techniques
- **Comprehensive Metrics**: Track efficiency, performance, and statistical fit simultaneously

---

## 2. Metrics to Track

Based on the project README, we need to measure:

### A. Computational Efficiency (Primary Focus)
| Metric | Description | Unit | When Measured |
|--------|-------------|------|---------------|
| **Inference Time** | Average time per sample | milliseconds (ms) | After training, on test set |
| **Training Time** | Time to complete one epoch | seconds (s) | During training, per epoch |
| **FLOPs** | Floating point operations | GFLOPs | One-time calculation |
| **Throughput** | Images processed per second | images/sec | During inference benchmark |
| **Peak Memory** | Maximum GPU memory used | MB | During training and inference |

### B. Predictive Performance
| Metric | Description | When Measured |
|--------|-------------|---------------|
| **Top-1 Accuracy** | Percentage of correct predictions | Training (per epoch) and Test (final) |
| **Precision** | TP / (TP + FP) per class | Test set only |
| **Recall** | TP / (TP + FN) per class | Test set only |
| **F1-Score** | Harmonic mean of precision and recall | Test set only |

### C. Statistical Model Fit
| Metric | Formula | Purpose |
|--------|---------|---------|
| **Log-Likelihood** | $\hat{\mathcal{L}} = \sum_i \log p(y_i \| x_i; \theta)$ | Foundation for AIC/BIC |
| **AIC** | $\text{AIC} = 2k - 2\hat{\mathcal{L}}$ | Trade-off between fit and complexity |
| **BIC** | $\text{BIC} = k \ln(n) - 2\hat{\mathcal{L}}$ | More aggressive complexity penalty |

Where:
- $k$ = number of parameters
- $n$ = number of training samples
- $\hat{\mathcal{L}}$ = log-likelihood on test set

### D. Model Complexity
| Metric | Description |
|--------|-------------|
| **Parameter Count** | Total number of trainable parameters |

---

## 3. Class Architecture

### 3.1 ModelEvaluator Class

```python
class ModelEvaluator:
    """
    Unified evaluator that trains, tests, and benchmarks any model architecture.
    Collects all metrics defined in the project README.

    Usage:
        evaluator = ModelEvaluator(dataset='mnist', batch_size=32, epochs=10)
        results = evaluator.evaluate(model, model_name='baseline-vit')
    """

    def __init__(self,
                 dataset: str = 'mnist',
                 batch_size: int = 32,
                 epochs: int = 10,
                 learning_rate: float = 0.001,
                 device: str = 'cuda',
                 seed: int = 42):
        """
        Initialize evaluator with dataset and training configuration.

        Args:
            dataset: 'mnist' or 'cifar10'
            batch_size: Batch size for training and testing
            epochs: Number of training epochs
            learning_rate: Learning rate for Adam optimizer
            device: 'cuda' or 'cpu'
            seed: Random seed for reproducibility
        """
        # Set random seeds for reproducibility
        # Load and prepare datasets (train and test loaders)
        # Store configuration

    def evaluate(self, model: nn.Module, model_name: str = 'model') -> dict:
        """
        Complete evaluation pipeline: train -> test -> benchmark -> compute stats.

        Args:
            model: PyTorch model instance (nn.Module)
            model_name: Human-readable name for results tracking

        Returns:
            Dictionary containing all metrics organized by category
        """
        # Initialize results structure
        # Count parameters
        # Run training phase
        # Run testing phase
        # Benchmark inference
        # Compute statistical metrics
        # Return comprehensive results

    def _train(self, model: nn.Module, results: dict) -> None:
        """
        Training loop with metric collection.

        Collects per-epoch:
        - Training loss
        - Training accuracy
        - Training time (seconds)
        - Peak GPU memory (MB)

        Updates results['training'] in-place.
        """
        # Setup optimizer (Adam) and loss function (CrossEntropyLoss)
        # For each epoch:
        #   - Reset timer and memory stats
        #   - Training loop over batches
        #   - Calculate epoch metrics
        #   - Store in results['training']

    def _test(self, model: nn.Module, results: dict) -> None:
        """
        Detailed testing with classification metrics.

        Collects:
        - Test accuracy (top-1)
        - Per-class precision, recall, F1-score
        - Confusion matrix (optional, for analysis)

        Updates results['testing'] in-place.
        """
        # Set model to eval mode
        # Collect predictions and ground truth
        # Compute accuracy
        # Compute precision, recall, F1 per class using sklearn
        # Store in results['testing']

    def _benchmark_inference(self, model: nn.Module, results: dict) -> None:
        """
        Inference speed and efficiency benchmarking.

        Collects:
        - Total inference time on test set
        - Average time per sample (ms)
        - Throughput (images/sec)
        - Peak GPU memory during inference (MB)

        Updates results['inference'] in-place.
        """
        # Warmup: run a few batches to stabilize GPU
        # Reset memory stats
        # Synchronize GPU (if using CUDA)
        # Start timer
        # Run inference on entire test set
        # Synchronize GPU
        # Stop timer
        # Calculate metrics: throughput, latency per sample
        # Store in results['inference']

    def _compute_statistical_metrics(self, model: nn.Module, results: dict) -> None:
        """
        Compute log-likelihood, AIC, and BIC.

        Log-likelihood: Sum of log probabilities of correct classes
        AIC: 2k - 2*log_likelihood
        BIC: k*ln(n) - 2*log_likelihood

        where k = parameter count, n = number of test samples

        Updates results['statistical'] in-place.
        """
        # Set model to eval mode
        # Get softmax probabilities for test set
        # Extract probabilities of correct classes
        # Compute log-likelihood: sum of log(correct_probs)
        # Get parameter count from results
        # Get dataset size
        # Calculate AIC and BIC
        # Store in results['statistical']

    def _count_parameters(self, model: nn.Module) -> int:
        """Count total trainable parameters."""
        # Sum p.numel() for all parameters where p.requires_grad

    def _estimate_flops(self, model: nn.Module) -> float:
        """
        Estimate FLOPs for the model (optional, may need external library).

        Can use libraries like:
        - fvcore (Facebook)
        - thop (torch-ops)
        - Or manual calculation based on architecture

        Returns FLOPs in billions (GFLOPs)
        """
        # TODO: Implement using one of the above methods
        # For now, can return None and add later
```

### 3.2 Results Structure

The `evaluate()` method returns a nested dictionary:

```python
results = {
    'model_name': 'baseline-vit',
    'config': {
        'dataset': 'mnist',
        'batch_size': 32,
        'epochs': 10,
        'learning_rate': 0.001
    },
    'model_complexity': {
        'parameters': 123456,
        'flops_giga': 1.23  # Optional
    },
    'training': {
        'loss_per_epoch': [2.3, 1.8, 1.4, ...],  # List of length epochs
        'accuracy_per_epoch': [45.2, 67.8, 78.3, ...],
        'time_per_epoch': [12.5, 12.3, 12.4, ...],  # seconds
        'peak_memory_mb': [256.7, 258.1, 257.9, ...],
        'final_train_accuracy': 98.5,
        'total_training_time': 125.4  # seconds
    },
    'testing': {
        'accuracy': 97.8,  # Top-1 accuracy
        'precision_per_class': [0.98, 0.97, 0.99, ...],  # 10 values for MNIST
        'recall_per_class': [0.97, 0.98, 0.96, ...],
        'f1_per_class': [0.97, 0.97, 0.97, ...],
        'macro_precision': 0.978,
        'macro_recall': 0.976,
        'macro_f1': 0.977
    },
    'inference': {
        'total_time': 2.34,  # seconds
        'avg_time_per_sample': 0.234,  # milliseconds
        'throughput': 4274,  # images/sec
        'peak_memory_mb': 145.6
    },
    'statistical': {
        'log_likelihood': -1234.56,
        'aic': 2500.12,
        'bic': 2678.45
    }
}
```

---

## 4. Model Interface Requirements

Any model to be evaluated must:

1. **Inherit from `nn.Module`**
2. **Implement `forward(x)` method** that:
   - Takes input tensor `x` of shape `(batch, channels, height, width)`
   - Returns logits of shape `(batch, num_classes)`
   - Does NOT apply softmax (handled by loss function)
3. **Accept standard initialization parameters** (for consistency):
   - `image_size`: Input image dimensions
   - `patch_size`: Patch size for tokenization
   - `num_classes`: Number of output classes
   - `dim`: Embedding dimension
   - `depth`: Number of transformer layers
   - `heads`: Number of attention heads
   - `mlp_dim`: MLP hidden dimension
   - `dropout`: Dropout probability

Example:
```python
class BaselineViT(nn.Module):
    def __init__(self, image_size=28, patch_size=4, num_classes=10,
                 dim=64, depth=6, heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        # Architecture definition

    def forward(self, x):
        # x: (batch, 1, 28, 28) for MNIST
        # Returns: (batch, 10) logits
        return logits
```

---

## 5. Usage Examples

### Example 1: Evaluate Baseline ViT on MNIST

```python
from models.baseline_vit import BaselineViT
from evaluation.evaluator import ModelEvaluator

# Create evaluator
evaluator = ModelEvaluator(
    dataset='mnist',
    batch_size=32,
    epochs=10,
    learning_rate=0.001,
    device='cuda'
)

# Instantiate model
model = BaselineViT(
    image_size=28,
    patch_size=4,
    num_classes=10,
    dim=64,
    depth=6,
    heads=4,
    mlp_dim=128,
    dropout=0.1
)

# Run full evaluation
results = evaluator.evaluate(model, model_name='baseline-vit-mnist')

# Print summary
print(f"Model: {results['model_name']}")
print(f"Parameters: {results['model_complexity']['parameters']:,}")
print(f"Test Accuracy: {results['testing']['accuracy']:.2f}%")
print(f"Inference Time: {results['inference']['avg_time_per_sample']:.3f} ms")
print(f"AIC: {results['statistical']['aic']:.2f}")
```

### Example 2: Compare Multiple Models

```python
from models.baseline_vit import BaselineViT
from models.performer_favor import PerformerFAVOR
from models.performer_relu import PerformerReLU

evaluator = ModelEvaluator(dataset='mnist', epochs=10)

# Evaluate all models
models = [
    (BaselineViT(), 'baseline-vit'),
    (PerformerFAVOR(), 'performer-favor'),
    (PerformerReLU(), 'performer-relu')
]

all_results = []
for model, name in models:
    results = evaluator.evaluate(model, model_name=name)
    all_results.append(results)

# Compare results
comparison_df = create_comparison_table(all_results)
print(comparison_df)
```

### Example 3: CIFAR-10 Evaluation

```python
# For CIFAR-10, adjust image_size and channels
evaluator = ModelEvaluator(dataset='cifar10', batch_size=64, epochs=20)

model = BaselineViT(
    image_size=32,  # CIFAR-10 is 32x32
    patch_size=4,
    num_classes=10,
    dim=128,  # Larger model for more complex dataset
    depth=8,
    heads=8,
    mlp_dim=256
)

results = evaluator.evaluate(model, model_name='baseline-vit-cifar10')
```

---

## 6. Implementation Notes

### 6.1 GPU Synchronization
When measuring time on CUDA:
```python
if torch.cuda.is_available():
    torch.cuda.synchronize()  # Before starting timer
start_time = time.time()
# ... operations ...
if torch.cuda.is_available():
    torch.cuda.synchronize()  # Before stopping timer
end_time = time.time()
```

### 6.2 Memory Tracking
```python
# Reset stats before measurement
torch.cuda.reset_peak_memory_stats()

# ... operations ...

# Get peak memory
peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
```

### 6.3 Inference Warmup
Run a few batches before timing to stabilize GPU:
```python
# Warmup
for i, (images, _) in enumerate(test_loader):
    if i >= 3:  # 3 warmup batches
        break
    _ = model(images.to(device))
```

### 6.4 Log-Likelihood Computation
```python
# Get softmax probabilities
log_probs = F.log_softmax(logits, dim=1)

# Extract log probabilities of correct classes
correct_log_probs = log_probs[range(len(labels)), labels]

# Sum for total log-likelihood
log_likelihood = correct_log_probs.sum().item()
```

### 6.5 Precision, Recall, F1 Calculation
Use sklearn for easy per-class metrics:
```python
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true=all_labels,
    y_pred=all_predictions,
    average=None  # Returns per-class metrics
)

# Also compute macro averages
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_true=all_labels,
    y_pred=all_predictions,
    average='macro'
)
```

### 6.6 Random Seed Management
For reproducibility:
```python
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 7. Directory Structure (Proposed)

```
efficient-rpe-vit/
├── DESIGN.md              # This file - framework design doc
├── README.md              # Project overview
├── baseline-vit.ipynb     # Original notebook (reference)
│
├── models/                # Model architectures
│   ├── __init__.py
│   ├── baseline_vit.py    # Standard ViT (quadratic attention)
│   ├── performer_favor.py # Performer with FAVOR+ (to implement)
│   ├── performer_relu.py  # Performer with ReLU (to implement)
│   └── attention/         # Attention mechanism modules
│       ├── __init__.py
│       ├── standard.py    # Quadratic softmax attention
│       ├── favor.py       # FAVOR+ linear attention
│       └── relu.py        # ReLU-based linear attention
│
├── evaluation/            # Evaluation framework
│   ├── __init__.py
│   ├── evaluator.py       # ModelEvaluator class
│   └── metrics.py         # Helper functions for metrics
│
├── data/                  # Dataset handling
│   ├── __init__.py
│   └── datasets.py        # MNIST and CIFAR-10 loaders
│
├── utils/                 # Utilities
│   ├── __init__.py
│   ├── visualization.py   # Plotting results
│   └── comparison.py      # Compare multiple model results
│
└── experiments/           # Experiment scripts
    ├── train_baseline.py  # Example: train baseline on MNIST
    └── compare_all.py     # Run all 9 models and compare
```

---

## 8. Next Steps

### Phase 1: Foundation (Start Here)
1. ✅ Create DESIGN.md (this file)
2. Create directory structure
3. Extract baseline ViT from notebook into `models/baseline_vit.py`
4. Implement `ModelEvaluator` class in `evaluation/evaluator.py`
5. Implement dataset loaders in `data/datasets.py`

### Phase 2: Validation
6. Test evaluator with baseline ViT on MNIST
7. Verify all metrics are collected correctly
8. Compare notebook results vs framework results (should match)

### Phase 3: Extension
9. Implement Performer attention mechanisms
10. Add RPE implementations
11. Create comparison utilities
12. Run full 9-model comparison

### Open Questions
- **FLOPs Calculation**: Which library to use? (fvcore, thop, or manual)
- **Results Storage**: Save to JSON, CSV, or database?
- **Visualization**: What plots do we need? (accuracy curves, speed comparison bars, etc.)
- **Hyperparameter Tuning**: Do we need a config system? (YAML files, argparse, etc.)

---

## 9. Dependencies (Estimated)

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0  # For precision, recall, F1
```

Optional:
```
fvcore  # For FLOPs calculation
tqdm    # Progress bars
pandas  # Results table management
seaborn # Advanced plotting
```

---

**Last Updated**: 2025-11-02
**Status**: Design phase - implementation pending
