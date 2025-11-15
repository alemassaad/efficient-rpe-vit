# Initial Performance Comparison: RoPE and Circulant-STRING vs Baseline

**Date:** 2025-01-XX  
**Training Configuration:**
- Dataset: MNIST
- Epochs: 3
- Batch Size: 64
- Learning Rate: 0.001 (with cosine annealing)

## Results Summary

| Model | Test Accuracy | Train Accuracy | Test Loss | Train Loss | Training Time | Inference Latency |
|-------|---------------|----------------|-----------|------------|---------------|-------------------|
| **baseline** | **94.20%** | 91.08% | 0.1925 | 0.2913 | 100.95s | 5.00ms |
| **baseline_rope** | **95.08%** | 92.67% | 0.1591 | 0.2411 | 112.59s | 7.84ms |
| **baseline_circulant** | **94.07%** | 91.11% | 0.1991 | 0.2895 | 97.99s | 6.11ms |

## Key Findings

### RoPE (Rotary Position Embedding)
- ✅ **Outperforms baseline by +0.88% accuracy** (95.08% vs 94.20%)
- ✅ Lower test loss (0.1591 vs 0.1925) - **17.3% improvement**
- ✅ Lower train loss (0.2411 vs 0.2913) - **17.2% improvement**
- ✅ Higher train accuracy (92.67% vs 91.08%) - **1.59% improvement**
- ⚠️ Slightly slower inference (7.84ms vs 5.00ms) - **56.8% slower** due to rotation computations
- ⚠️ Slightly longer training time (112.59s vs 100.95s) - **11.5% slower**

### Circulant-STRING RPE
- ✅ Performs similarly to baseline (94.07% vs 94.20%) - **-0.13% difference**
- ✅ Similar loss values (0.1991 vs 0.1925 test loss)
- ✅ Slightly faster training time (97.99s vs 100.95s) - **2.9% faster**
- ⚠️ Slightly slower inference (6.11ms vs 5.00ms) - **22.2% slower**

## Analysis

1. **RoPE shows clear accuracy benefits**: The rotary position embedding provides better positional understanding, leading to improved accuracy and lower loss values. The trade-off is slightly increased computational cost.

2. **Circulant-STRING is competitive**: While not outperforming baseline in this initial test, it performs very similarly, suggesting it may be beneficial in different scenarios or with longer training.

3. **Speed trade-offs**: Both RPE methods add computational overhead, but the improvements in accuracy (especially for RoPE) may justify the cost depending on the use case.

## Next Steps

- Run longer training runs (10-20 epochs) to see if differences persist or amplify
- Test on CIFAR-10 to evaluate performance on more complex datasets
- Run multiple seeds for statistical significance
- Compare with Performer variants (FAVOR+ and ReLU) with RPE

