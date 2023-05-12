import torch
import torch.nn as nn
from typing import Optional
import math


# https://github.com/microsoft/unilm/blob/840b020ce5397c1c0dd9e62a2f4c66fba3afd934/beats/backbone.py
class RelativePositionalBias(nn.Module):
    def __init__(self, num_heads, q_head_dim, gated=False, num_buckets=320, max_distance=800, gate_dim=8):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.num_heads = num_heads
        self.q_head_dim = q_head_dim
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        # only for gated option
        if gated:
            self.grep_linear = nn.Linear(q_head_dim, gate_dim)
            self.grep_w = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.gated = gated
        assert gate_dim % 2 == 0, "gate_dim must be even"
        self.gate_dim = gate_dim

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions >
                                 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = - \
                torch.min(relative_positions,
                          torch.zeros_like(relative_positions))

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
            torch.log(relative_positions.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(
                relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small,
                                        relative_positions, relative_postion_if_large)
        return relative_buckets

    # TODO: use unidirectional for masked transformer
    def compute_bias(self, query_length, key_length):
        context_position = torch.arange(
            query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(
            relative_position,
            bidirectional=True
        )
        relative_position_bucket = relative_position_bucket.to(
            self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def forward(self, q, position_bias=None):
        # TODO: assumes source sequence length == target sequence length
        # B, num_heads, N, C // num_heads
        bsz, num_heads, seq_len, q_head_dim = q.size()
        assert num_heads == self.num_heads, "num_heads must be equal to self.num_heads"
        assert q_head_dim == self.q_head_dim, "q_head_dim must be equal to self.q_head_dim"

        # setting up position_bias once per model forward pass
        if position_bias is None:
            position_bias = self.compute_bias(seq_len, seq_len)
            position_bias = position_bias.unsqueeze(0).repeat(
                bsz, 1, 1, 1).view(bsz * self.num_heads, seq_len, seq_len)

        attn_mask_rel_pos = position_bias

        if self.gated:
            query_layer = q.view(bsz, self.num_heads, seq_len, self.q_head_dim)
            _B, _H, _L, __ = query_layer.size()
            gate_a, gate_b = torch.sigmoid(self.grep_linear(query_layer).view(
                _B, _H, _L, 2, self.gate_dim//2).sum(-1, keepdim=False)).chunk(2, dim=-1)
            # scale a*b, bias -(a+2) for w (self.grep_a)
            gate_a_1 = gate_a * (gate_b * self.grep_w - 1.0) + 2.0
            attn_mask_rel_pos = gate_a_1.view(
                bsz * self.num_heads, seq_len, 1) * position_bias

        return attn_mask_rel_pos, position_bias
    
class RelativePositional2DBias(nn.Module):
    """Adds two independent bias terms for row and column.
    
    Parameters:
        grid: (row, col) tuple of the grid size
    """
    def __init__(self, grid, num_heads, q_head_dim, gated=False, num_buckets=(5, 70), max_distance=(5, 70), gate_dim=(8,8)):
        """    
        """
        super().__init__()
        self.grid = grid

        self.h_rel_pos_bias = RelativePositionalBias(
            num_heads, q_head_dim, gated, num_buckets[0], max_distance[0], gate_dim[0])
        self.w_rel_pos_bias = RelativePositionalBias(
            num_heads, q_head_dim, gated, num_buckets[1], max_distance[1], gate_dim[1])
       
        self.num_heads = num_heads
        self.q_head_dim = q_head_dim

    def forward(self, q, position_bias=None):
        # TODO: assumes source sequence length == target sequence length
        # B, num_heads, N, C // num_heads
        bsz, num_heads, seq_len, q_head_dim = q.size()
        assert num_heads == self.num_heads, "num_heads must be equal to self.num_heads"
        assert q_head_dim == self.q_head_dim, "q_head_dim must be equal to self.q_head_dim"

        # setting up position_bias once per model forward pass
        if position_bias is None:
            position_bias = None, None
        
        attn_mask_rel_pos_0, position_bias_0 = self.h_rel_pos_bias(q, position_bias[0])
        attn_mask_rel_pos_1, position_bias_1 = self.w_rel_pos_bias(q, position_bias[1])

        attn_mask_rel_pos = attn_mask_rel_pos_0 + attn_mask_rel_pos_1
        position_bias = position_bias_0, position_bias_1

        return attn_mask_rel_pos, position_bias


class RelativePositionalBiasFactory:
    def __init__(self, **kwargs):
        if "style" in kwargs and kwargs["style"] == "2d":
            kwargs.pop("style")
            self.cls = RelativePositional2DBias
        else:
            self.cls = RelativePositionalBias
        self.kwargs = kwargs
    
    def __call__(self, *args, **kwargs):
        return self.cls(*args, **{**kwargs, **self.kwargs})