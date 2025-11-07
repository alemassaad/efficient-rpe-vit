# Efficient-RPE-ViT: Improving Performer Vision Transformers with Relative Positional Encodings

## Project Objective

This project investigates the downstream accuracy improvement of Vision Transformers (ViTs) utilizing Performer architectures through integration with various Relative Positional Encoding (RPE) methods. The study examines small ViT models with Performer backbones for attention computation, considering two Performer variants: (a) models leveraging positive random features for unbiased approximation of the softmax kernel (FAVOR+), and (b) Performer-ReLU architectures. The Performer-ViT models are enriched with three RPE mechanisms: (1) kernelized attention with relative positional encoding [Luo et al., 2021], (2) circulant-STRING [Schenck et al., 2025], and (3) rotary position embedding (RoPE) [Su et al., 2024]. Efficient implementations of these RPE-enriched Performers are provided and compared against standard brute-force attention ViT. The central research question examines whether RPE integration can effectively close the accuracy gap between standard ViT and Performer variants. Experimental validation is conducted on MNIST and CIFAR-10 datasets, with comprehensive comparison of training time, inference time, and classification accuracy across all model variants.

## Core Technical Challenge

The primary technical challenge involves implementing RPE mechanisms within the linear $\mathcal{O}(N)$ attention framework of the Performer architecture. Standard RPE methods operate on the quadratic $\mathbf{Q K}^\top$ attention matrix, requiring adaptation for efficient integration with kernelized attention mechanisms. This work focuses on developing efficient RPE-Performer fusion approaches that preserve the computational advantages of linear attention while incorporating the positional information encoded by RPE methods.

## Model Architectures

The experimental framework encompasses twelve model variants combining three attention mechanisms with four positional encoding approaches.

| Category | Attention Mechanism | RPE Mechanism | Complexity |
| :--- | :--- | :--- | :--- |
| **Baseline (Quadratic)** | Brute-Force Softmax Attention | None (Absolute PE) | $\mathcal{O}(N^2)$ |
| | Brute-Force Softmax Attention | Kernelized RPE [Luo et al., 2021] | $\mathcal{O}(N^2)$ |
| | Brute-Force Softmax Attention | Circulant-STRING [Schenck et al., 2025] | $\mathcal{O}(N^2)$ |
| | Brute-Force Softmax Attention | RoPE [Su et al., 2024] | $\mathcal{O}(N^2)$ |
| **Performer-FAVOR+** | Positive Random Features | None (Absolute PE) | $\mathcal{O}(N)$ |
| | Positive Random Features | Kernelized RPE [Luo et al., 2021] | $\mathcal{O}(N \log N)$ |
| | Positive Random Features | Circulant-STRING [Schenck et al., 2025] | $\mathcal{O}(N)$ |
| | Positive Random Features | RoPE [Su et al., 2024] | $\mathcal{O}(N)$ |
| **Performer-ReLU** | ReLU Kernel Approximation | None (Absolute PE) | $\mathcal{O}(N)$ |
| | ReLU Kernel Approximation | Kernelized RPE [Luo et al., 2021] | $\mathcal{O}(N \log N)$ |
| | ReLU Kernel Approximation | Circulant-STRING [Schenck et al., 2025] | $\mathcal{O}(N)$ |
| | ReLU Kernel Approximation | RoPE [Su et al., 2024] | $\mathcal{O}(N)$ |

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

# OR use the automated init script (macOS/Linux only):
source .claude/init.sh
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
python experiments/train_baseline.py --dataset mnist

# Train with full outputs (checkpoints, metrics, plots)
python experiments/train_baseline.py --dataset mnist \
    --save-model \
    --save-metrics \
    --plot \
    --save-plots

# Train on CIFAR-10
python experiments/train_baseline.py --dataset cifar10 --epochs 20
```

### Running Tests
```bash
# Test model architecture
python test_model.py

# Test training functionality
python test_training.py
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

