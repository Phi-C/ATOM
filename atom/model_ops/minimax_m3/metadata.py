from dataclasses import dataclass

import torch


@dataclass
class MiniMaxM3SparsePrefillMetadata:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int
    total_kv_blocks: int


@dataclass
class MiniMaxM3SparseDecodeMetadata:
    seq_lens: torch.Tensor
    block_table: torch.Tensor


@dataclass
class MiniMaxM3SparseMetadata:
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    prefill: MiniMaxM3SparsePrefillMetadata | None = None
    decode: MiniMaxM3SparseDecodeMetadata | None = None
