import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from flash_attn import flash_attn_varlen_qkvpacked, flash_attn_varlen_kvpacked
from flash_attn.bert_padding import pad_input, unpad_input


class MoBAAttention(nn.Module):
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
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = query.size()
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        num_blocks = math.ceil(seq_len / self.block_size)
        
        pad_len = (num_blocks * self.block_size) - seq_len
        if pad_len > 0:
            k_pad = F.pad(k, (0, 0, 0, pad_len))
            v_pad = F.pad(v, (0, 0, 0, pad_len))
        else:
            k_pad = k
            v_pad = v
        
        k_blocks = k_pad.view(batch_size, self.num_heads, num_blocks, self.block_size, self.head_dim)
        v_blocks = v_pad.view(batch_size, self.num_heads, num_blocks, self.block_size, self.head_dim)
        
        k_mean = k_blocks.mean(dim=3)
        
        q_reshaped = q.unsqueeze(3)
        scores = torch.matmul(q_reshaped, k_mean.transpose(-1, -2)).squeeze(3)
        
        positions = torch.arange(seq_len, device=q.device)
        query_block_indices = positions // self.block_size
        
        block_indices = torch.arange(num_blocks, device=q.device)
        causal_mask = query_block_indices.unsqueeze(1) < block_indices.unsqueeze(0)
        
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        current_block_mask = (query_block_indices.unsqueeze(1) == block_indices.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(current_block_mask, float('-inf'))
        
        effective_top_k = min(self.top_k, num_blocks)
        _, topk_indices = torch.topk(scores, k=effective_top_k, dim=-1)
        
        outputs = []
        
        for b in range(batch_size):
            self_attn_outputs = []
            moba_attn_outputs = []
            
            for h in range(self.num_heads):
                cu_seqlens = []
                current_idx = 0
                
                q_self_list = []
                k_self_list = []
                v_self_list = []
                
                q_moba_list = []
                k_moba_list = []
                v_moba_list = []
                
                for i in range(seq_len):
                    current_block_idx = i // self.block_size
                    current_block_start = current_block_idx * self.block_size
                    current_block_end = min((current_block_idx + 1) * self.block_size, seq_len)
                    
                    q_self = q[b, h, i].unsqueeze(0)
                    k_self = k[b, h, current_block_start:i+1]
                    v_self = v[b, h, current_block_start:i+1]
                    
                    q_self_list.append(q_self)
                    k_self_list.append(k_self)
                    v_self_list.append(v_self)
                    
                    cu_seqlens.append(current_idx)
                    current_idx += k_self.size(0)
                    
                    selected_blocks = topk_indices[b, h, i]
                    for block_idx in selected_blocks:
                        block_idx = block_idx.item()
                        if block_idx == current_block_idx or block_idx > current_block_idx:
                            continue
                            
                        block_start = block_idx * self.block_size
                        block_end = min((block_idx + 1) * self.block_size, seq_len)
                        
                        q_moba = q[b, h, i].unsqueeze(0)
                        k_moba = k[b, h, block_start:block_end]
                        v_moba = v[b, h, block_start:block_end]
                        
                        q_moba_list.append(q_moba)
                        k_moba_list.append(k_moba)
                        v_moba_list.append(v_moba)
                        
                cu_seqlens.append(current_idx)
                cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=q.device)
                
                if q_self_list:
                    q_self_cat = torch.cat(q_self_list, dim=0)
                    k_self_cat = torch.cat(k_self_list, dim=0)
                    v_self_cat = torch.cat(v_self_list, dim=0)
                    
                    qkv_self = torch.cat([q_self_cat.unsqueeze(2), k_self_cat.unsqueeze(2), v_self_cat.unsqueeze(2)], dim=2)
                    self_out = flash_attn_varlen_qkvpacked(
                        qkv_self, cu_seqlens_tensor, 
                        max_seqlen=max(1, cu_seqlens[-1] - cu_seqlens[0]), 
                        dropout_p=0.0, 
                        causal=True
                    )
                    
                    self_attn_outputs.append(self_out)
                else:
                    self_attn_outputs.append(torch.zeros(seq_len, self.head_dim, device=q.device))
                
                if q_moba_list:
                    q_moba_cat = torch.cat(q_moba_list, dim=0)
                    k_moba_cat = torch.cat(k_moba_list, dim=0)
                    v_moba_cat = torch.cat(v_moba_list, dim=0)
                    
                    moba_out = flash_attn_varlen_kvpacked(
                        q_moba_cat, torch.cat([k_moba_cat, v_moba_cat], dim=-1),
                        cu_seqlens_tensor[:-1], cu_seqlens_tensor[:-1],
                        max_seqlen=max(1, q_moba_cat.size(0)),
                        dropout_p=0.0,
                        causal=False
                    )
                    
                    moba_attn_outputs.append(moba_out)
                else:
                    moba_attn_outputs.append(torch.zeros(seq_len, self.head_dim, device=q.device))
            
            self_outputs = torch.stack(self_attn_outputs, dim=0)
            moba_outputs = torch.stack(moba_attn_outputs, dim=0)
            
            combined_output = self_outputs + moba_outputs
            outputs.append(combined_output)
        
        output = torch.stack(outputs, dim=0)
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
        
        self.self_attn = MoBAAttention(
            d_model=d_model,
            num_heads=num_heads,
            block_size=block_size,
            top_k=top_k,
            dropout=dropout,
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_output = self.self_attn(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            attn_mask=attn_mask,
        )
        x = x + self.dropout(attn_output)
        
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class MoBATransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        block_size: int = 512,
        top_k: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.layers = nn.ModuleList([
            MoBATransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                block_size=block_size,
                top_k=top_k,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        x = self.norm(x)
        
        return self.output_layer(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class HybridMoBATransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        block_size: int,
        top_k: int,
        dropout: float = 0.1,
        use_moba: bool = True,
    ):
        super().__init__()
        
        self.use_moba = use_moba
        
        if use_moba:
            self.self_attn = MoBAAttention(
                d_model=d_model,
                num_heads=num_heads,
                block_size=block_size,
                top_k=top_k,
                dropout=dropout,
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_x = self.norm1(x)
        
        if self.use_moba:
            attn_output = self.self_attn(
                query=norm_x,
                key=norm_x,
                value=norm_x,
                attn_mask=attn_mask,
            )
        else:
            if attn_mask is not None:
                attn_mask = attn_mask.squeeze(0).squeeze(0)
                
            attn_output, _ = self.self_attn(
                query=norm_x,
                key=norm_x,
                value=norm_x,
                attn_mask=attn_mask,
            )
        
        x = x + self.dropout(attn_output)
        
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        
        return x


class LayerWiseHybridMoBATransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        num_full_attn_layers: int = 3,
        d_ff: int = 3072,
        block_size: int = 4096,
        top_k: int = 12,
        dropout: float = 0.1,
        max_seq_len: int = 1024 * 1024,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
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
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dtype == torch.long:
            x = self.embedding(x) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)
            x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        x = self.norm(x)
        
        return self.output_layer(x)


def create_moba_1m_model(pretrained_model_path=None):
    model = LayerWiseHybridMoBATransformer(
        vocab_size=32000,
        d_model=2048,
        num_heads=32,
        num_layers=32,
        num_full_attn_layers=3,
        d_ff=8192,
        block_size=4096,
        top_k=12,
        dropout=0.0,
        max_seq_len=1024 * 1024,
    )
    
    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path))
        
    return model


if __name__ == "__main__":
    model = LayerWiseHybridMoBATransformer(
        vocab_size=10000,
        d_model=256,
        num_heads=4,
        num_layers=4,
        num_full_attn_layers=1,
        d_ff=1024,
        block_size=128,
        top_k=2,
        max_seq_len=4096,
    )
    
    batch_size = 2
    seq_len = 512
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_ids, mask)
    
    print(f"Output shape: {output.shape}")