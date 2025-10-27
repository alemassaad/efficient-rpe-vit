# Efficient-RPE-ViT: Improving Performer Vision Transformers with Relative Positional Encodings

## 🚀 Project Goal

The objective of this project is to address the accuracy gap between standard Vision Transformers (ViT) and highly efficient **Performer-based ViTs** (which use linear attention) by integrating various **Relative Positional Encodings (RPEs)**. We devise and test efficient RPE implementations specifically tailored for the linear Performer attention mechanism.

## 🧠 Core Challenge: RPE-Performer Fusion

The most significant technical challenge is implementing RPEs within the **linear $\mathcal{O}(N)$ attention** of the Performer. Standard RPEs operate on the quadratic $\mathbf{Q K}^\top$ attention matrix. Our work will focus on creating **efficient RPE-Performer fusions** that maintain the computational gains of the Performer while incorporating the relational context provided by RPEs.

## 🛠️ Models to Implement and Compare (9 Total)

The project requires building and testing three main types of ViT models across two Performer backbone variants.

| Category | Attention Mechanism (Backbone) | RPE Mechanism |
| :--- | :--- | :--- | :--- |
| **I. Baseline (Quadratic)** | Regular **Brute-Force Softmax Attention** ($\mathcal{O}(N^2)$) | None (Standard PE) |
| **II. Efficient Baselines** | **(a) Performer-FAVOR+** (Positive Random Features) | None |
| | **(b) Performer-ReLU** | None |
| **III. RPE-Enriched (a)** | Performer-FAVOR+ | **(1) Most General RPE** [Luo et al., 2021] |
| | Performer-FAVOR+ | **(2) circulant-STRING** [Schenck et al., 2025] |
| | Performer-FAVOR+ | **(3) Regular RoPE** [Su et al., 2024] |
| **IV. RPE-Enriched (b)** | Performer-ReLU | **(1) Most General RPE** [Luo et al., 2021] |
| | Performer-ReLU | **(2) circulant-STRING** [Schenck et al., 2025] |
| | Performer-ReLU | **(3) Regular RoPE** [Su et al., 2024] |

## 🧪 Experimental Roadmap

### A. Datasets
All models will be trained and evaluated on:
1.  **MNIST**
2.  **CIFAR-10**

### B. Evaluation Metrics
We will compare all 10 model variants across a comprehensive set of accuracy, fit, and efficiency metrics.

| Focus Area | Key Metrics to Measure | Purpose |
| :--- | :--- | :--- |
| **Computational Efficiency (Primary Focus)** | **Inference Time** (Average per Sample) | Measure real-world speed/latency. |
| | **Training Time** (Per Epoch) | Evaluate overall training cost. |
| | **FLOPs** (Floating Point Operations) | Theoretical computational complexity. |
| **Predictive Performance** | **Top-1 Accuracy**, Precision, Recall, F1-Score | Standard deep learning classification results. |
| **Statistical Model Fit** | **Log-Likelihood** ($\hat{\mathcal{L}}$) | Foundation for fit criteria. |
| | **AIC** (Akaike Information Criterion) | Assess trade-off between fit and complexity. |
| | **BIC** (Bayesian Information Criterion) | Penalizes complexity more aggressively than AIC. |
| **Model Complexity** | **Model Size** (Number of Parameters, $k$) | Parameter count. |

## 💡 Expected Outcome

The final goal is to determine if RPEs can effectively **close the accuracy gap** between the standard ViT and the highly efficient Performer-ViT variants, validating the approach with rigorous testing of both speed and performance.
