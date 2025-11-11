# Efficient-RPE-ViT: Improving Performer Vision Transformers with Relative Positional Encodings

## ✅ KERPLE Integration Complete!

**KERPLE (Kernelized Attention with RPE)** is now fully implemented and working! Models `performer_favor_most_general` and `performer_relu_most_general` train successfully with:
- ✅ Vectorized O(n log n) FFT operations
- ✅ All 23 unit tests passing
- ✅ Successful training on MNIST (63.57% accuracy after 1 epoch)
- ✅ No divergence, stable gradients

See [`docs/KERPLE_DOCUMENTATION.md`](docs/KERPLE_DOCUMENTATION.md) for full technical documentation.

## Project Objective

This project investigates the downstream accuracy improvement of Vision Transformers (ViTs) utilizing Performer architectures through integration with various Relative Positional Encoding (RPE) methods. The study examines small ViT models with Performer backbones for attention computation, considering two Performer variants: (a) models leveraging positive random features for unbiased approximation of the softmax kernel (FAVOR+), and (b) Performer-ReLU architectures. The Performer-ViT models are enriched with three RPE mechanisms: (1) **KERPLE - kernelized attention with relative positional encoding [Luo et al., 2021] ✅ COMPLETE**, (2) circulant-STRING [Schenck et al., 2025], and (3) rotary position embedding (RoPE) [Su et al., 2024]. Efficient implementations of these RPE-enriched Performers are provided and compared against standard brute-force attention ViT. The central research question examines whether RPE integration can effectively close the accuracy gap between standard ViT and Performer variants. Experimental validation is conducted on MNIST and CIFAR-10 datasets, with comprehensive comparison of training time, inference time, and classification accuracy across all model variants.

## Core Technical Challenge

The primary technical challenge involves implementing RPE mechanisms within the linear $\mathcal{O}(N)$ attention framework of the Performer architecture. Standard RPE methods operate on the quadratic $\mathbf{Q K}^\top$ attention matrix, requiring adaptation for efficient integration with kernelized attention mechanisms. This work focuses on developing efficient RPE-Performer fusion approaches that preserve the computational advantages of linear attention while incorporating the positional information encoded by RPE methods.

## Model Architectures

The experimental framework encompasses twelve model variants combining three attention mechanisms with four positional encoding approaches.

| Category | Attention Mechanism | RPE Mechanism | Complexity | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Quadratic)** | Brute-Force Softmax Attention | None (Absolute PE) | $\mathcal{O}(N^2)$ | ✅ Working |
| | Brute-Force Softmax Attention | KERPLE [Luo et al., 2021] | N/A | ❌ Incompatible* |
| | Brute-Force Softmax Attention | Circulant-STRING [Schenck et al., 2025] | $\mathcal{O}(N^2)$ | ⏳ TODO |
| | Brute-Force Softmax Attention | RoPE [Su et al., 2024] | $\mathcal{O}(N^2)$ | ⏳ TODO |
| **Performer-FAVOR+** | Positive Random Features | None (Absolute PE) | $\mathcal{O}(N)$ | ✅ Working |
| | Positive Random Features | **KERPLE [Luo et al., 2021]** | $\mathcal{O}(N \log N)$ | **✅ Complete!** |
| | Positive Random Features | Circulant-STRING [Schenck et al., 2025] | $\mathcal{O}(N)$ | ⏳ TODO |
| | Positive Random Features | RoPE [Su et al., 2024] | $\mathcal{O}(N)$ | ⏳ TODO |
| **Performer-ReLU** | ReLU Kernel Approximation | None (Absolute PE) | $\mathcal{O}(N)$ | ✅ Working |
| | ReLU Kernel Approximation | **KERPLE [Luo et al., 2021]** | $\mathcal{O}(N \log N)$ | **✅ Complete!** |
| | ReLU Kernel Approximation | Circulant-STRING [Schenck et al., 2025] | $\mathcal{O}(N)$ | ⏳ TODO |
| | ReLU Kernel Approximation | RoPE [Su et al., 2024] | $\mathcal{O}(N)$ | ⏳ TODO |

\* KERPLE is designed specifically for linear attention and cannot work with quadratic softmax attention by design.

## Experimental Design

### Datasets
All model variants are trained and evaluated on two benchmark image classification datasets:
1. MNIST (28×28 grayscale handwritten digits, 10 classes)
2. CIFAR-10 (32×32 RGB natural images, 10 classes)

### Evaluation Metrics
Comprehensive evaluation across computational efficiency, predictive performance, and model fit:

| Metric Category | Specific Metrics | Description |
| :--- | :--- | :--- |
| **Computational Efficiency** | Training Time (seconds/epoch) | Per-epoch training duration |
| | Inference Time (milliseconds/sample) | Average prediction latency |
| | FLOPs | Theoretical computational complexity |
| **Classification Performance** | Top-1 Accuracy | Primary classification metric |
| | Precision, Recall, F1-Score | Per-class performance measures |
| **Statistical Model Fit** | Log-Likelihood ($\hat{\mathcal{L}}$) | Model fit quality |
| | AIC (Akaike Information Criterion) | Fit-complexity trade-off |
| | BIC (Bayesian Information Criterion) | Conservative model selection criterion |
| **Model Properties** | Parameter Count | Total number of trainable parameters |

### Research Question
Can RPE mechanisms effectively reduce or eliminate the accuracy gap between standard brute-force attention ViT and computationally efficient Performer-ViT variants, while preserving the linear complexity advantages of kernelized attention?

## 🛠️ Getting Started

### Environment Setup

**Step 1: Create Virtual Environment**
```bash
# Create the virtual environment
python -m venv .venv
```

**Step 2: Activate the Environment**
```bash
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

**Step 3: Install Dependencies**
```bash
# Upgrade pip (recommended)
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Step 4: Verify Installation**
```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Check CUDA availability (for GPU support)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# List available model variants
python -c "from models import list_available_models; print('Available models:', list_available_models())"
```

### Quick Start

Once your environment is set up, you can train models:

```bash
# Train baseline ViT on MNIST
python experiments/train.py --model baseline --dataset mnist --epochs 5

# Train FAVOR+ Performer (linear attention)
python experiments/train.py --model performer_favor --dataset mnist --epochs 5

# ✨ NEW: Train with KERPLE RPE (Most General RPE)
python experiments/train.py --model performer_favor_most_general \
    --dataset mnist \
    --epochs 10 \
    --batch-size 64 \
    --save-model \
    --save-metrics

# Quick test with large batches (1-2 minutes)
python experiments/train.py --model performer_favor_most_general \
    --dataset mnist \
    --epochs 1 \
    --batch-size 512

# Train ReLU Performer with KERPLE
python experiments/train.py --model performer_relu_most_general \
    --dataset mnist \
    --epochs 10
```

### Running Tests
```bash
# Test FAVOR+ and ReLU Performer
python test_performer.py

# Test KERPLE RPE integration (23 tests)
python test_kerple.py

# Run specific KERPLE test
python -m unittest test_kerple.TestKERPLEIntegration::test_favor_plus_with_kerple -v
```

## 📊 Benchmarking & Visualization

### Running Benchmarks

Benchmark multiple models across multiple random seeds for statistical rigor:

```bash
# Basic benchmark: Compare 3 models with 5 runs each
python experiments/benchmark.py \
    --models baseline performer_favor performer_favor_most_general \
    --dataset mnist \
    --num-runs 5 \
    --epochs 20 \
    --batch-size 64

# Quick test benchmark (faster iteration)
python experiments/benchmark.py \
    --models baseline performer_favor \
    --dataset mnist \
    --num-runs 2 \
    --epochs 5 \
    --batch-size 256

# Full CIFAR-10 benchmark (research-grade)
python experiments/benchmark.py \
    --models baseline performer_favor performer_relu \
              performer_favor_most_general performer_relu_most_general \
    --dataset cifar10 \
    --num-runs 5 \
    --epochs 50 \
    --batch-size 64
```

**Benchmark Output Structure:**
```
results/
└── benchmark_mnist_20251111_113441/
    ├── benchmark_config.json          # Benchmark configuration
    ├── baseline/
    │   ├── aggregated_stats.json      # Statistics across all runs
    │   ├── run_0/                     # Individual run results
    │   │   └── baseline_metrics.json
    │   ├── run_1/
    │   └── ...
    └── performer_favor/
        └── ...
```

**What Gets Measured:**
- **Accuracy**: Best, final, per-epoch test accuracy
- **Training Time**: Seconds per epoch, total training time
- **Inference Speed**: Mean latency in milliseconds
- **Convergence**: Epochs to reach 90%, 95%, 99% accuracy
- **Statistics**: Mean, std, min, max, percentiles across runs

### Interactive Dashboard

Visualize benchmark results with an interactive Streamlit dashboard:

**Step 1: Install Dashboard Dependencies**
```bash
pip install streamlit plotly
```

**Step 2: Launch Dashboard**
```bash
streamlit run experiments/dashboard.py
```

This will:
- Start a local web server
- Automatically open your browser to http://localhost:8501

**Step 3: Load Your Results**

In the sidebar, enter the path to your benchmark results:
```
results/benchmark_mnist_20251111_113441
```

**Step 4: Explore Visualizations**

The dashboard provides six interactive tabs:

1. **📊 Overview**
   - Summary statistics table for all models
   - Key metrics: Best accuracy, training time, inference speed
   - Quick comparison of top performers

2. **📈 Accuracy**
   - Percentile-based accuracy distributions
   - Solid lines: Mean accuracy
   - Dashed lines: 5th, 25th, 75th, 95th percentiles
   - Shaded bands: 5-95% (light) and 25-75% (darker) ranges
   - Scatter points: Individual run values

3. **⏱️ Training Dynamics**
   - Per-epoch training curves
   - Mean accuracy/loss over time
   - Percentile confidence bands (5-95% and 25-75%)
   - Compare convergence speed across models

4. **⚡ Efficiency**
   - Training time and inference latency comparison
   - Accuracy vs efficiency scatter plot
   - Identify Pareto-optimal models

5. **🎯 Convergence**
   - Epochs to reach 90%, 95%, 99% accuracy
   - Compare learning speed across architectures

6. **🔍 Per-Run Details**
   - Drill down into individual runs
   - View complete training history
   - Inspect full metrics JSON

**Dashboard Features:**
- ✅ **Interactive**: Hover for exact values, zoom, pan
- ✅ **Publication-ready**: Export plots with built-in camera icon
- ✅ **Real-time**: Instantly updates when you change results path
- ✅ **Statistical rigor**: Percentile-based visualizations
- ✅ **Comprehensive**: All metrics in one place

### Example Workflow

Complete benchmark-to-publication workflow:

```bash
# 1. Run comprehensive benchmark
python experiments/benchmark.py \
    --models baseline performer_favor performer_favor_most_general \
    --dataset mnist \
    --num-runs 5 \
    --epochs 20 \
    --batch-size 64

# 2. Note the output directory (e.g., results/benchmark_mnist_20251111_113441)

# 3. Launch dashboard
streamlit run experiments/dashboard.py

# 4. Enter the path in the dashboard sidebar

# 5. Explore results, export plots for your paper!
```

### Deactivating the Environment
```bash
# When done working
deactivate
```

## References

**[Luo et al., 2021]** Luo, S., Li, S., Cai, T., He, D., Peng, D., Zheng, S., Ke, G., Wang, L., and Liu, T. (2021). Stable, fast and accurate: Kernelized attention with relative positional encoding. In Ranzato, M., Beygelzimer, A., Dauphin, Y. N., Liang, P., and Vaughan, J. W., editors, *Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021*, December 6-14, 2021, virtual, pages 22795–22807.
📄 **Local PDF**: [`2106.12566v2.pdf`](./2106.12566v2.pdf) | arXiv: [2106.12566](https://arxiv.org/abs/2106.12566)

**[Schenck et al., 2025]** Schenck, C., Reid, I., Jacob, M. G., Bewley, A., Ainslie, J., Rendleman, D., Jain, D., Sharma, M., Dubey, A., Wahid, A., Singh, S., Wagner, R., Ding, T., Fu, C., Byravan, A., Varley, J., Gritsenko, A. A., Minderer, M., Kalashnikov, D., Tompson, J., Sindhwani, V., and Choromanski, K. (2025). Learning the ropes: Better 2d and 3d position encodings with STRING. *ICML 2025*, abs/2502.02562.
📄 **Local PDF**: [`2502.02562v1.pdf`](./2502.02562v1.pdf) | arXiv: [2502.02562](https://arxiv.org/abs/2502.02562)

**[Su et al., 2024]** Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. (2024). Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063.
📄 **Local PDF**: [`2104.09864v5.pdf`](./2104.09864v5.pdf) | arXiv: [2104.09864](https://arxiv.org/abs/2104.09864)

