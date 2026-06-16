from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

SPARSE_BLOCK_SIZE = 128


@dataclass
class MiniMaxM3SGLangMetadata:
    """Per-forward SGLang metadata for MiniMax-M3 sparse attention."""

    is_decode: bool
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    max_seq_len: int
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    max_query_len: int = 1
    total_kv_blocks: int = 0


def validate_minimax_m3_page_size(page_size: int) -> None:
    """MiniMax-M3 sparse blocks must line up 1:1 with SGLang KV pages."""

    if int(page_size) != SPARSE_BLOCK_SIZE:
        raise ValueError(
            "MiniMax-M3 sparse attention requires SGLang page size 128 "
            f"(got {page_size}). Launch with --page-size 128."
        )


def _get_batch_size(forward_batch) -> int:
    return int(getattr(forward_batch, "batch_size"))


def _slice_batch_tensor(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    return tensor[:batch_size].to(dtype=torch.int32)


def _get_query_lens(forward_batch, batch_size: int) -> torch.Tensor:
    query_lens = getattr(forward_batch, "extend_seq_lens", None)
    if query_lens is None:
        query_lens = getattr(forward_batch, "seq_lens")
    return _slice_batch_tensor(query_lens, batch_size)


def _get_prefix_lens(
    forward_batch,
    batch_size: int,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
) -> torch.Tensor:
    prefix_lens = getattr(forward_batch, "extend_prefix_lens", None)
    if prefix_lens is None:
        return (seq_lens - query_lens).to(dtype=torch.int32)
    return _slice_batch_tensor(prefix_lens, batch_size)


def build_minimax_m3_block_table(forward_batch, page_size: int) -> torch.Tensor:
    """Build a physical block table from SGLang's request-token table."""

    validate_minimax_m3_page_size(page_size)
    batch_size = _get_batch_size(forward_batch)
    req_pool_indices = forward_batch.req_pool_indices[:batch_size]
    req_to_token = forward_batch.req_to_token_pool.req_to_token
    token_table = req_to_token[req_pool_indices, :].clone()

    if not forward_batch.forward_mode.is_decode_or_idle():
        query_lens = _get_query_lens(forward_batch, batch_size)
        seq_lens = _slice_batch_tensor(forward_batch.seq_lens, batch_size)
        prefix_lens = _get_prefix_lens(forward_batch, batch_size, seq_lens, query_lens)
        offset = 0
        out_cache_loc = forward_batch.out_cache_loc
        for req_idx in range(batch_size):
            prefix_len = int(prefix_lens[req_idx].item())
            query_len = int(query_lens[req_idx].item())
            if query_len > 0:
                token_table[req_idx, prefix_len : prefix_len + query_len] = (
                    out_cache_loc[offset : offset + query_len]
                )
            offset += query_len

    max_seq_len = int(_slice_batch_tensor(forward_batch.seq_lens, batch_size).max().item())
    max_blocks = (max_seq_len + page_size - 1) // page_size
    block_table = token_table[:, : max_blocks * page_size : page_size] // page_size
    return block_table.to(dtype=torch.int32).contiguous()


def build_minimax_m3_forward_metadata(
    forward_batch,
    block_table: torch.Tensor,
    page_size: int,
) -> MiniMaxM3SGLangMetadata:
    """Translate SGLang ForwardBatch fields into MiniMax-M3 sparse metadata."""

    validate_minimax_m3_page_size(page_size)
    batch_size = _get_batch_size(forward_batch)
    seq_lens = _slice_batch_tensor(forward_batch.seq_lens, batch_size)
    max_seq_len = int(seq_lens.max().item()) if batch_size else 0

    if forward_batch.forward_mode.is_decode_or_idle():
        return MiniMaxM3SGLangMetadata(
            is_decode=True,
            seq_lens=seq_lens,
            block_table=block_table,
            max_seq_len=max_seq_len,
        )

    query_lens = _get_query_lens(forward_batch, batch_size)
    context_lens = _get_prefix_lens(forward_batch, batch_size, seq_lens, query_lens)
    cu_seqlens_q = torch.empty(batch_size + 1, dtype=torch.int32, device=seq_lens.device)
    cu_seqlens_k = torch.empty(batch_size + 1, dtype=torch.int32, device=seq_lens.device)
    cu_seqlens_q[0] = 0
    cu_seqlens_k[0] = 0
    torch.cumsum(query_lens, dim=0, out=cu_seqlens_q[1:])
    torch.cumsum(seq_lens, dim=0, out=cu_seqlens_k[1:])
    total_kv_blocks = int(
        torch.div(seq_lens + page_size - 1, page_size, rounding_mode="floor")
        .sum()
        .item()
    )

    return MiniMaxM3SGLangMetadata(
        is_decode=False,
        seq_lens=seq_lens,
        block_table=block_table,
        max_seq_len=max_seq_len,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        context_lens=context_lens,
        max_query_len=int(query_lens.max().item()) if batch_size else 0,
        total_kv_blocks=total_kv_blocks,
    )


def _get_page_size(forward_batch) -> int:
    return int(getattr(forward_batch.token_to_kv_pool, "page_size", 1))


def _ensure_side_caches(
    layer,
    forward_batch,
    key: torch.Tensor,
    index_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    page_size = _get_page_size(forward_batch)
    validate_minimax_m3_page_size(page_size)

    k_buffer, _ = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    num_slots = int(k_buffer.shape[0])
    num_blocks = num_slots // page_size
    if num_blocks <= 0:
        raise RuntimeError("MiniMax-M3 sparse attention received an empty KV pool.")

    main_shape = (
        num_blocks,
        2,
        page_size,
        layer.num_kv_heads,
        layer.head_dim,
    )
    index_shape = (num_blocks, page_size, layer.idx_head_dim)

    main_cache = getattr(layer, "_sglang_m3_main_cache", None)
    if (
        main_cache is None
        or main_cache.shape != main_shape
        or main_cache.device != key.device
        or main_cache.dtype != key.dtype
    ):
        main_cache = torch.empty(main_shape, dtype=key.dtype, device=key.device)
        layer._sglang_m3_main_cache = main_cache

    index_cache = getattr(layer, "_sglang_m3_index_cache", None)
    if (
        index_cache is None
        or index_cache.shape != index_shape
        or index_cache.device != index_key.device
        or index_cache.dtype != index_key.dtype
    ):
        index_cache = torch.empty(index_shape, dtype=index_key.dtype, device=index_key.device)
        layer._sglang_m3_index_cache = index_cache

    return main_cache, index_cache


def _insert_sparse_cache(
    layer,
    forward_batch,
    key: torch.Tensor,
    value: torch.Tensor,
    index_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    main_cache, index_cache = _ensure_side_caches(layer, forward_batch, key, index_key)
    page_size = _get_page_size(forward_batch)
    slot_mapping = forward_batch.out_cache_loc[: key.shape[0]].to(dtype=torch.long)
    valid = slot_mapping >= 0

    slots = slot_mapping[valid]
    block_ids = torch.div(slots, page_size, rounding_mode="floor")
    block_offsets = slots % page_size

    key = key.view(-1, layer.num_kv_heads, layer.head_dim)[valid]
    value = value.view(-1, layer.num_kv_heads, layer.head_dim)[valid]
    index_key = index_key.view(-1, layer.idx_head_dim)[valid]
    main_cache[block_ids, 0, block_offsets] = key.to(main_cache.dtype)
    main_cache[block_ids, 1, block_offsets] = value.to(main_cache.dtype)
    index_cache[block_ids, block_offsets] = index_key.to(index_cache.dtype)
    return main_cache, index_cache


def minimax_m3_sparse_attention_for_sglang(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    layer,
    forward_batch=None,
    save_kv_cache: bool = True,
) -> torch.Tensor:
    """Run MiniMax-M3 lightning-indexer sparse attention for SGLang plugin mode."""

    if forward_batch is None:
        from atom.plugin.sglang.runtime import get_current_forward_batch

        forward_batch = get_current_forward_batch()
    if forward_batch is None:
        raise RuntimeError("MiniMax-M3 sparse attention requires a SGLang ForwardBatch.")

    page_size = _get_page_size(forward_batch)
    validate_minimax_m3_page_size(page_size)
    if save_kv_cache:
        main_cache, index_cache = _insert_sparse_cache(
            layer,
            forward_batch,
            key,
            value,
            index_key,
        )
    else:
        main_cache, index_cache = _ensure_side_caches(layer, forward_batch, key, index_key)

    block_table = build_minimax_m3_block_table(forward_batch, page_size)
    metadata = build_minimax_m3_forward_metadata(forward_batch, block_table, page_size)

    q = query.view(-1, layer.num_heads, layer.head_dim)
    iq = index_query.view(-1, layer.num_idx_heads, layer.idx_head_dim)
    output = torch.empty_like(q)

    from atom.model_ops.minimax_m3.index_topk import (
        minimax_m3_index_topk,
        minimax_m3_index_topk_decode,
    )
    from atom.model_ops.minimax_m3.sparse_attn import (
        minimax_m3_sparse_attn,
        minimax_m3_sparse_attn_decode,
    )

    if metadata.is_decode:
        batch_size = metadata.seq_lens.shape[0]
        topk_idx = minimax_m3_index_topk_decode(
            iq[:batch_size],
            index_cache,
            metadata.block_table,
            metadata.seq_lens,
            metadata.max_seq_len,
            layer.topk_blocks,
            layer.init_blocks,
            layer.local_blocks,
            layer.num_kv_heads,
            layer.scaling,
        )
        minimax_m3_sparse_attn_decode(
            q[:batch_size],
            main_cache,
            topk_idx,
            metadata.block_table,
            metadata.seq_lens,
            layer.num_kv_heads,
            layer.scaling,
            output[:batch_size],
        )
        if batch_size < output.shape[0]:
            output[batch_size:].zero_()
    else:
        assert metadata.cu_seqlens_q is not None
        assert metadata.cu_seqlens_k is not None
        assert metadata.context_lens is not None
        num_tokens = int(metadata.cu_seqlens_q[-1].item())
        topk_idx = minimax_m3_index_topk(
            iq[:num_tokens],
            index_cache,
            metadata.block_table,
            metadata.cu_seqlens_q,
            metadata.seq_lens,
            metadata.context_lens,
            metadata.max_query_len,
            metadata.max_seq_len,
            layer.topk_blocks,
            layer.init_blocks,
            layer.local_blocks,
            layer.num_kv_heads,
            layer.scaling,
        )
        minimax_m3_sparse_attn(
            q[:num_tokens],
            main_cache,
            topk_idx,
            metadata.block_table,
            metadata.cu_seqlens_q,
            metadata.seq_lens,
            metadata.context_lens,
            metadata.max_query_len,
            layer.num_kv_heads,
            layer.scaling,
            output[:num_tokens],
        )
        if num_tokens < output.shape[0]:
            output[num_tokens:].zero_()

    return output.reshape_as(query)
