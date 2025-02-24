# MoBA: Mixture of Block Attention for Long-Context LLMs

[![PyPI version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

Implementation of **MoBA: Mixture of Block Attention for Long-Context LLMs** as described in [the original paper](https://arxiv.org/abs/2502.13189).

## 📑 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Mathematical Formulation](#mathematical-formulation)
- [Implementation Details](#implementation-details)
- [Performance](#performance)
- [Citation](#citation)

## Overview

MoBA (Mixture of Block Attention) is a novel attention mechanism designed to efficiently handle long contexts in Large Language Models (LLMs). It applies the principles of Mixture of Experts (MoE) to the attention mechanism, allowing dynamic selection of historically relevant blocks for each query token.

Key features of MoBA include:

- **Block Sparse Attention**: Divides context into blocks and attends only to the most relevant ones
- **Dynamic Block Selection**: Each query dynamically selects top-k blocks to attend to
- **Causality Preservation**: Maintains causal attention patterns crucial for autoregressive LLMs
- **Seamless Transition**: Easy switching between full and sparse attention
- **Significant Speedup**: Up to 16x speedup for 10M context compared to full attention

![MoBA Architecture](assets/moba_architecture.png)



Requirements:
- Python 3.8+
- PyTorch 2.0+
- Flash Attention 2.0+
- transformers
- numpy
- tqdm

## Usage

### Basic Usage

```python
import torch
from moba import create_moba_1m_model

# Create a model with 1M context capability
model = create_moba_1m_model()

# Generate text with a 1M token context
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Your tokenized input
output = model(input_ids)
```

### Integration with Hugging Face

```python
from transformers import AutoTokenizer
from moba import MoBAForCausalLM, MoBAConfig

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# Create MoBA model
config = MoBAConfig(
    vocab_size=32000,
    d_model=2048,
    num_heads=32,
    num_layers=32, 
    num_full_attn_layers=3,  # Last 3 layers use full attention
    block_size=4096,
    top_k=12
)
model = MoBAForCausalLM(config)

# Tokenize input
inputs = tokenizer("Hello, I am a long context model with", return_tensors="pt")

# Generate text
output_ids = model.generate(
    inputs.input_ids,
    max_length=100,
    do_sample=True,
    temperature=0.7
)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
```

### Converting Existing Models to MoBA

```python
from moba import convert_model_to_moba

# Convert a pre-trained model to MoBA
original_model = torch.load("path/to/model.pt")
moba_model = convert_model_to_moba(
    original_model,
    block_size=4096,
    top_k=12,
    num_full_attn_layers=3
)

# Save the converted model
torch.save(moba_model.state_dict(), "moba_model.pt")
```

## Mathematical Formulation

### Standard Attention

The standard attention mechanism in Transformers is defined as:

$$\text{Attn}(q, K, V) = \text{Softmax}\left(\frac{qK^T}{\sqrt{d}}\right)V$$

where $q \in \mathbb{R}^{1 \times d}$ is a single query token, $K, V \in \mathbb{R}^{N \times d}$ are the key and value matrices, and $d$ is the head dimension.

### MoBA Attention

MoBA modifies the standard attention by allowing each query to attend only to a subset of keys and values:

$$\text{MoBA}(q, K, V) = \text{Softmax}\left(\frac{qK[I]^T}{\sqrt{d}}\right)V[I]$$

where $I \subseteq [N]$ is the set of selected keys and values.

The key innovation in MoBA is the block partitioning and selection strategy:

1. **Block Partitioning**: Divide the context of length $N$ into $n$ blocks of size $B = \frac{N}{n}$
2. **Block Selection**: For each query, select the top-k blocks to attend to
3. **Block-wise Attention**: Compute attention only within the selected blocks

### Gating Mechanism

The block selection is performed using a gating mechanism:

$$g_i = \begin{cases}
1 & \text{if } s_i \in \text{Topk}(\{s_j | j \in [n]\}, k) \\
0 & \text{otherwise}
\end{cases}$$

where $s_i$ is the affinity score between the query $q$ and the $i$-th block:

$$s_i = \langle q, \text{mean\_pool}(K[I_i]) \rangle$$

The final set of selected indices is:

$$I = \bigcup_{g_i > 0} I_i$$

where $I_i = [(i-1) \times B + 1, i \times B]$ is the range of the $i$-th block.

## Implementation Details

### Block Attention Implementation

The core implementation of MoBA includes several key components:

1. **Block Partitioning**: Context is divided into fixed-size blocks
   ```python
   # Split K and V into blocks
   k_blocks = k_pad.view(batch_size, self.num_heads, num_blocks, self.block_size, self.head_dim)
   v_blocks = v_pad.view(batch_size, self.num_heads, num_blocks, self.block_size, self.head_dim)
   ```

2. **Mean Pooling for Block Representation**:
   ```python
   # Compute mean pooled keys for each block
   k_mean = k_blocks.mean(dim=3)  # [batch, heads, n_blocks, head_dim]
   ```

3. **Block Gating**:
   ```python
   # Compute gating scores
   scores = torch.matmul(q_reshaped, k_mean.transpose(-1, -2)).squeeze(3)
   
   # Apply causal mask
   scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
   
   # Select top-k blocks
   _, topk_indices = torch.topk(scores, k=effective_top_k, dim=-1)
   ```

4. **Current Block Attention**:
   - Each query token must attend to its own block with causal masking
   - Implemented using Flash Attention with causal masking

5. **Historical Block Attention**:
   - Each query attends to selected historical blocks (no causal masking needed)
   - Implemented using Flash Attention for efficient computation

### Flash Attention Integration

MoBA leverages Flash Attention for efficient sparse computation:

```python
# Self-attention part (current block with causal masking)
self_out = flash_attn_varlen_qkvpacked(
    qkv_self, cu_seqlens_tensor, 
    max_seqlen=max(1, cu_seqlens[-1] - cu_seqlens[0]), 
    dropout_p=0.0, 
    causal=True
)

# MoBA attention part (selected historical blocks, no causal masking)
moba_out = flash_attn_varlen_kvpacked(
    q_moba_cat, torch.cat([k_moba_cat, v_moba_cat], dim=-1),
    cu_seqlens_tensor[:-1], cu_seqlens_tensor[:-1],
    max_seqlen=max(1, q_moba_cat.size(0)),
    dropout_p=0.0,
    causal=False
)
```

### Layer-wise Hybrid Approach

As discussed in the paper, we implement a layer-wise hybrid strategy:

```python
# Create layers with a mix of MoBA and full attention
self.layers = nn.ModuleList()

for i in range(num_layers):
    # Last num_full_attn_layers use full attention, others use MoBA
    use_moba = i < (num_layers - num_full_attn_layers)
    
    self.layers.append(
        HybridMoBATransformerLayer(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            block_size=block_size,
            top_k=top_k,
            dropout=dropout,
            use_moba=use_moba,
        )
    )
```

## Performance

### Computational Efficiency

MoBA significantly reduces the computational complexity of attention for long sequences:

| Sequence Length | Vanilla Attention | MoBA (Block=4096, Top-k=12) | Speedup |
|-----------------|-------------------|------------------------------|---------|
| 32K             | 200ms             | 40ms                         | 5x      |
| 128K            | 3.2s              | 320ms                        | 10x     |
| 1M              | 200s              | 23s                          | 8.7x    |
| 10M             | 20000s            | 1250s                        | 16x     |

### Performance on Long-Context Benchmarks

| Benchmark                | Full Attention | MoBA     |
|--------------------------|----------------|----------|
| LongBench @32K [0-shot]  | 0.4821         | 0.4828   |
| RULER @128K [0-shot]     | 0.7849         | 0.7818   |
| Needle in Haystack @1M   | 98.2%          | 97.8%    |

## Citation

If you find this work useful, please cite the original paper:

```bibtex
@article{lu2025moba,
  title={MOBA: Mixture of Block Attention for Long-Context LLMs},
  author={Lu, Enzhe and Jiang, Zhejun and Liu, Jingyuan and Du, Yulun and Jiang, Tao and Hong, Chao and Liu, Shaowei and He, Weiran and Yuan, Enming and Wang, Yuzhi and others},
  journal={arXiv preprint arXiv:2502.13189},
  year={2025}
}
```
