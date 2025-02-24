import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from flash_attn import flash_attn_varlen_qkvpacked, flash_attn_varlen_kvpacked
from flash_attn.bert_padding import unpad_input, pad_input


class MoBAGating(nn.Module):
    def __init__(self, d_model, num_heads, block_size, top_k):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.top_k = top_k
        
    def forward(self, q, k, seq_len):
        batch_size = q.size(0)
        num_blocks = math.ceil(seq_len / self.block_size)
        
        pad_len = (num_blocks * self.block_size) - seq_len
        if pad_len > 0:
            k_pad = F.pad(k, (0, 0, 0, pad_len))
        else:
            k_pad = k
            
        k_blocks = k_pad.view(batch_size, self.num_heads, num_blocks, self.block_size, self.head_dim)
        k_mean = k_blocks.mean(dim=3)
        
        q_reshaped = q.unsqueeze(3)
        scores = torch.matmul(q_reshaped, k_mean.transpose(-1, -2)).squeeze(3)
        
        positions = torch.arange(seq_len, device=q.device)
        query_block_indices = positions // self.block_size
        
        block_indices = torch.arange(num_blocks, device=q.device)
        causal_mask = query_block_indices.unsqueeze(1) < block_indices.unsqueeze(0)
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        effective_top_k = min(self.top_k, num_blocks)
        _, topk_indices = torch.topk(scores, k=effective_top_k, dim=-1)
        
        return topk_indices, query_block_indices


class BlockSelector(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, q, k, v, topk_indices, query_block_indices, block_size, seq_len):
        batch_size, num_heads = q.size(0), q.size(1)
        
        self_kv_list = []
        moba_kv_list = []
        
        for b in range(batch_size):
            for h in range(num_heads):
                for i in range(seq_len):
                    current_block_idx = i // block_size
                    current_block_start = current_block_idx * block_size
                    current_block_end = min((current_block_idx + 1) * block_size, seq_len)
                    
                    self_kv_list.append((
                        b, h, i,
                        current_block_start, min(i+1, current_block_end)
                    ))
                    
                    selected_blocks = topk_indices[b, h, i]
                    for block_idx in selected_blocks:
                        block_idx = block_idx.item()
                        if block_idx == current_block_idx or block_idx > current_block_idx:
                            continue
                            
                        block_start = block_idx * block_size
                        block_end = min((block_idx + 1) * block_size, seq_len)
                        
                        moba_kv_list.append((
                            b, h, i,
                            block_start, block_end
                        ))
        
        return self_kv_list, moba_kv_list
class OptimizedMoBAAttention(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        block_size: int,
        top_k: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.top_k = top_k
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.gating = MoBAGating(d_model, num_heads, block_size, top_k)
        self.block_selector = BlockSelector()
    
    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.size()
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        topk_indices, query_block_indices = self.gating(q, k, seq_len)
        
        self_kv_list, moba_kv_list = self.block_selector(
            q, k, v, topk_indices, query_block_indices, self.block_size, seq_len
        )
        
        output = torch.zeros_like(q)
        
        for batch_idx, head_idx, query_idx, key_start, key_end in self_kv_list:
            if key_start < key_end:
                q_vec = q[batch_idx, head_idx, query_idx:query_idx+1]
                k_vec = k[batch_idx, head_idx, key_start:key_end]
                v_vec = v[batch_idx, head_idx, key_start:key_end]
                
                qkv = torch.cat([
                    q_vec.view(-1, self.head_dim).unsqueeze(1),
                    k_vec.view(-1, self.head_dim).unsqueeze(1),
                    v_vec.view(-1, self.head_dim).unsqueeze(1)
                ], dim=1)
                
                cu_seqlens = torch.tensor([0, k_vec.size(0)], device=q.device, dtype=torch.int32)
                max_seqlen = k_vec.size(0)
                
                attn_out = flash_attn_varlen_qkvpacked(
                    qkv, cu_seqlens, max_seqlen=max_seqlen, dropout_p=0.0, causal=True
                )
                
                output[batch_idx, head_idx, query_idx] = attn_out[0]
        
        for batch_idx, head_idx, query_idx, key_start, key_end in moba_kv_list:
            q_vec = q[batch_idx, head_idx, query_idx:query_idx+1]
            k_vec = k[batch_idx, head_idx, key_start:key_end]
            v_vec = v[batch_idx, head_idx, key_start:key_end]
            
            kv = torch.cat([k_vec, v_vec], dim=-1)
            
            cu_seqlens_q = torch.tensor([0, 1], device=q.device, dtype=torch.int32)
            cu_seqlens_kv = torch.tensor([0, k_vec.size(0)], device=q.device, dtype=torch.int32)
            
            attn_out = flash_attn_varlen_kvpacked(
                q_vec, kv, cu_seqlens_q, cu_seqlens_kv,
                max_seqlen=k_vec.size(0), dropout_p=0.0, causal=False
            )
            
            output[batch_idx, head_idx, query_idx] += attn_out[0]
        
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output


class MoBATransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        block_size: int,
        top_k: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.self_attn = OptimizedMoBAAttention(
            d_model=d_model,
            num_heads=num_heads,
            block_size=block_size,
            top_k=top_k,
            dropout=dropout,
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        attn_output = self.self_attn(self.norm1(x), attn_mask)
        x = x + self.dropout(attn_output)
        
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x
class FullAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        norm_x = self.norm1(x)
        
        if attn_mask is not None:
            attn_mask = attn_mask.squeeze(0).squeeze(0)
            
        attn_output, _ = self.self_attn(norm_x, norm_x, norm_x, attn_mask=attn_mask)
        x = x + self.dropout(attn_output)
        
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024*1024):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LayerWiseHybridMoBATransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=2048,
        num_heads=32,
        num_layers=32,
        num_full_attn_layers=3,
        d_ff=8192,
        block_size=4096,
        top_k=12,
        dropout=0.1,
        max_seq_len=1024*1024,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers - num_full_attn_layers):
            self.layers.append(
                MoBATransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    block_size=block_size,
                    top_k=top_k,
                    dropout=dropout,
                )
            )
            
        for i in range(num_full_attn_layers):
            self.layers.append(
                FullAttentionLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
            )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        if input_ids.dim() == 2:
            batch_size, seq_len = input_ids.size()
            
            x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
            x = self.pos_encoding(x)
            x = self.dropout(x)
            
            for layer in self.layers:
                x = layer(x, attention_mask)
                
            x = self.norm(x)
            logits = self.lm_head(x)
            
            return logits
        else:
            x = input_ids
            
            for layer in self.layers:
                x = layer(x, attention_mask)
                
            x = self.norm(x)
            logits = self.lm_head(x)
            
            return logits


class MoBAConfig:
    def __init__(
        self,
        vocab_size=32000,
        d_model=2048,
        num_heads=32,
        num_layers=32,
        num_full_attn_layers=3,
        d_ff=8192,
        block_size=4096,
        top_k=12,
        dropout=0.0,
        max_seq_len=1024*1024,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_full_attn_layers = num_full_attn_layers
        self.d_ff = d_ff
        self.block_size = block_size
        self.top_k = top_k
        self.dropout = dropout
        self.max_seq_len = max_seq_len


def create_moba_1m_model(config=None, pretrained_path=None):
    if config is None:
        config = MoBAConfig()
        
    model = LayerWiseHybridMoBATransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_full_attn_layers=config.num_full_attn_layers,
        d_ff=config.d_ff,
        block_size=config.block_size,
        top_k=config.top_k,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
    )
    
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))
        
    return model


class MoBAForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = create_moba_1m_model(config)
        
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)
    
    def generate(
        self,
        input_ids,
        max_length=100,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.0,
        pad_token_id=None,
        eos_token_id=None,
        use_cache=True,
    ):
        batch_size, seq_len = input_ids.size()
        
        generated = input_ids
        past_key_values = None
        
        for _ in range(max_length):
            with torch.no_grad():
                inputs = generated
                outputs = self.model(inputs)
                
                next_token_logits = outputs[:, -1, :]
                
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for previous_token in generated[i]:
                            next_token_logits[i, previous_token] /= repetition_penalty
                
                if do_sample:
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float('Inf')
                    
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = -float('Inf')
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                if eos_token_id is not None:
                    next_tokens = next_tokens.masked_fill(
                        generated[:, -1] == eos_token_id, eos_token_id
                    )
                
                generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
                
                if eos_token_id is not None and (generated == eos_token_id).any(dim=1).all():
                    break
        
        return generated


if __name__ == "__main__":
    config = MoBAConfig(
        vocab_size=10000,
        d_model=512,
        num_heads=8,
        num_layers=6,
        num_full_attn_layers=2,
        d_ff=2048,
        block_size=128,
        top_k=3,
        dropout=0.1,
        max_seq_len=4096,
    )
    
    model = create_moba_1m_model(config)
    
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output shape: {logits.shape}")