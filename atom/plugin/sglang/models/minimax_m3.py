from __future__ import annotations

import torch
from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
from aiter.rotary_embedding import get_rope
from torch import nn
from transformers import PretrainedConfig

from atom.config import QuantizationConfig
from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.linear import MinimaxM3QKVParallelLinearWithIndexer, RowParallelLinear
from atom.plugin.sglang.attention_backend.minimax_m3_sparse import (
    minimax_m3_sparse_attention_for_sglang,
)


def _rope_theta(config: PretrainedConfig) -> float:
    return getattr(config, "rope_theta", 1000000.0)


class SGLangMiniMaxM3SparseAttention(nn.Module):
    """MiniMax-M3 sparse attention for ATOM's SGLang plugin path."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: str = "bf16",
    ) -> None:
        super().__init__()
        del cache_config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        if self.total_num_heads % self.tp_size != 0:
            raise ValueError("num_attention_heads must be divisible by TP size.")
        self.num_heads = self.total_num_heads // self.tp_size

        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.tp_size:
            if self.total_num_kv_heads % self.tp_size != 0:
                raise ValueError("num_key_value_heads must divide TP size.")
        elif self.tp_size % self.total_num_kv_heads != 0:
            raise ValueError("TP size must divide num_key_value_heads replication.")
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)

        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        sparse_cfg = config.sparse_attention_config
        self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
        self.num_idx_heads = self.num_kv_heads
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.index_q_size = self.num_idx_heads * self.idx_head_dim
        self.topk_blocks = int(sparse_cfg["sparse_topk_blocks"])
        self.sparse_block_size = int(sparse_cfg["sparse_block_size"])
        self.init_blocks = int(sparse_cfg.get("sparse_init_block", 0))
        self.local_blocks = int(sparse_cfg.get("sparse_local_block", 0))
        self.score_type = sparse_cfg.get("sparse_score_type", "max")
        if self.score_type != "max":
            raise ValueError(
                "MiniMax-M3 SGLang sparse attention only supports "
                f"score_type='max', got {self.score_type!r}."
            )

        self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            self.total_idx_heads,
            self.idx_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary_dim = int(self.head_dim * getattr(config, "partial_rotary_factor", 1.0))
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=_rope_theta(config),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.index_q_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
        self.index_k_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
        self.index_rotary_emb = self.rotary_emb

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        if isinstance(qkv, tuple):
            qkv = qkv[0]
        q, k, v, index_q, index_k = qkv.split(
            [
                self.q_size,
                self.kv_size,
                self.kv_size,
                self.index_q_size,
                self.idx_head_dim,
            ],
            dim=-1,
        )

        q = self.q_norm(
            q.reshape(*q.shape[:-1], self.num_heads, self.head_dim).contiguous()
        ).reshape(q.shape)
        k = self.k_norm(
            k.reshape(*k.shape[:-1], self.num_kv_heads, self.head_dim).contiguous()
        ).reshape(
            k.shape
        )
        q, k = self.rotary_emb(positions, q, k)

        index_q = self.index_q_norm(
            index_q.reshape(
                *index_q.shape[:-1], self.num_idx_heads, self.idx_head_dim
            ).contiguous()
        ).reshape(index_q.shape)
        index_k = self.index_k_norm(index_k)
        index_q, index_k = self.index_rotary_emb(positions, index_q, index_k)

        attn_output = minimax_m3_sparse_attention_for_sglang(
            q,
            k,
            v,
            index_q,
            index_k,
            self,
        )
        return self.o_proj(attn_output)
