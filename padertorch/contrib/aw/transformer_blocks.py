"""Implementation of Transformer blocks, Self-Attention, Multi-head attention, gated relative position bias, deep norm?"""

import torch
import torch.nn as nn
from typing import Optional
import math

from padertorch.modules.fully_connected import fully_connected_stack

# https://github.com/microsoft/unilm/blob/840b020ce5397c1c0dd9e62a2f4c66fba3afd934/beats/backbone.py


class RelativePositionalBias(nn.Module):
    def __init__(self, num_heads, q_head_dim, gated=False, num_buckets=320, max_distance=1280, gate_dim=8):
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

class RelativePositionalBiasFactory:
    def __init__(self, **kwargs):
        self.cls = RelativePositionalBias
        self.kwargs = kwargs
    
    def __call__(self, *args, **kwargs):
        return self.cls(*args, **{**kwargs, **self.kwargs})

class TorchMultiHeadAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=True,
                 add_bias_kv=False,
                 dropout=0.0,
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 use_k_bias=False,
                 rel_pos_bias_factory=False):
        super().__init__()
        self._attn = nn.MultiheadAttention(dim,
                                           num_heads,
                                           dropout=dropout,
                                           bias=qkv_bias,
                                           add_bias_kv=add_bias_kv,
                                           batch_first=True)
        self.in_proj_weight = self._attn.in_proj_weight
        self.out_proj = self._attn.out_proj
        self.scale = (dim // num_heads)**-0.5
        assert rel_pos_bias_factory is False, "rel_pos_bias must be False for TorchMultiHeadAttention"

    def forward(self, x, attn_mask=None, position_bias=None):
        assert position_bias is None, "position_bias must be None for StandardMultiHeadAttention"
        x, _ = self._attn(self.scale*x, x, x, attn_mask=attn_mask)
        return x

# TODO: improve architecture, avoid passing position_bias around
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=True,
                 dropout=0.0,
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 rel_pos_bias_factory: Optional[RelativePositionalBiasFactory] = None,
                 use_k_bias=True):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
            if use_k_bias:
                self.k_bias = nn.Parameter(torch.zeros(dim))
            else:
                self.k_bias = None
        else:
            self.q_bias = None
            self.v_bias = None
            self.k_bias = None

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        if rel_pos_bias_factory is not None:
            self.rel_pos_bias = rel_pos_bias_factory(num_heads=num_heads, q_head_dim=head_dim)
        else:
            self.rel_pos_bias = None

        # compatibility with TorchMultiHeadAttention
        self.in_proj_weight = self.qkv.weight
        self.out_proj = self.proj

    def forward(self, x, attn_mask=None, position_bias=None):
        B, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            if self.k_bias is not None:
                qkv_bias = torch.cat(
                    (
                        self.q_bias,
                        self.k_bias,
                        self.v_bias,
                    )
                )
            else:
                qkv_bias = torch.cat(
                    (
                        self.q_bias,
                        torch.zeros_like(self.v_bias, requires_grad=False),
                        self.v_bias,
                    )
                )
        qkv = nn.functional.linear(
            input=x, weight=self.qkv.weight, bias=qkv_bias)

        # reshape to 3, B, num_heads, N, C // num_heads (head dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)

        q = q * self.scale  # TODO: this is not in-place, unlike before
        alpha = 32  # TODO: what is this, taken from BEATs
        q *= 1 / alpha

        # attn shape: B, num_heads, N, N
        attn = (q @ k.transpose(-2, -1))

        attn = (attn - attn.max(dim=-1, keepdim=True)[0]) * alpha

        if attn_mask is not None:
            # attn_mask.shape == (N, N)
            attn += attn_mask

        if self.rel_pos_bias is not None:
            attn_bias, position_bias = self.rel_pos_bias(
                q*alpha/self.scale, position_bias)
            attn += attn_bias.view(B, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        if self.rel_pos_bias:
            return x, position_bias
        else:
            return x


class AttentionBlock(nn.Module):
    """"Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        qkv_bias (bool):  If True, add a learnable bias to q, k, v. Default: True
        dropout (float): Dropout rate. Default: 0.0
        attn_dropout (float): Attention dropout rate. Default: 0.0
        residual_scale (float): Scale factor for residual branch. Default: 1.0
        style (str): `pre-ln` or `post-ln` block structure. Default: `pre-ln`
        rel_pos_bias (RelativePositionalBias, optional): If not None, add relative positional bias to the self-attention layer. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        attn (nn.Module): Attention layer. Default: MultiHeadSelfAttention

    >>> embed_dim, num_heads = 512, 16
    >>> block = AttentionBlock(embed_dim, num_heads, rel_pos_bias=RelativePositionalBias(num_heads, q_head_dim=embed_dim//num_heads))
    >>> x = torch.randn(4, 128, 512)
    >>> out, position_bias = block(x, position_bias=None)
    >>> assert position_bias is not None
    >>> out.shape
    torch.Size([4, 128, 512])

    """

    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.0,
        attn_dropout=0.0,
        residual_scale=1.0,
        style="pre-ln",
        rel_pos_bias_factory: Optional[RelativePositionalBias] = None,
        act_layer='gelu',
        norm_layer=nn.LayerNorm,
        attn=TorchMultiHeadAttention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attn = attn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout=dropout,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            rel_pos_bias_factory=rel_pos_bias_factory,
        )

        self.use_rel_pos_bias = rel_pos_bias_factory is not None

        self.norm2 = norm_layer(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = fully_connected_stack(input_size=dim,
                                        hidden_size=[mlp_hidden_dim],
                                        output_size=dim,
                                        activation= act_layer,
                                        dropout=dropout)
        
        self.residual_scale = residual_scale
        self.style = style

    def pre_ln_forward(self, x, attn_mask=None, position_bias=None):
        if self.use_rel_pos_bias:
            x_attn, position_bias = self.attn(
                self.norm1(x), attn_mask, position_bias)
        else:
            x_attn = self.attn(self.norm1(x), attn_mask)

        x = self.residual_scale*x + x_attn
        x = self.residual_scale*x + self.mlp(self.norm2(x))

        if self.use_rel_pos_bias:
            return x, position_bias
        else:
            return x

    def post_ln_forward(self, x, attn_mask=None, position_bias=None):
        if self.use_rel_pos_bias:
            x_attn, position_bias = self.attn(x, attn_mask, position_bias)
        else:
            x_attn = self.attn(x, attn_mask)

        x = self.residual_scale*x + x_attn
        x = self.norm1(x)
        x = self.residual_scale*x + self.mlp(x)
        x = self.norm2(x)

        if self.use_rel_pos_bias:
            return x, position_bias
        else:
            return x

    def forward(self, x, attn_mask=None, position_bias=None):
        if self.style == "pre-ln":
            return self.pre_ln_forward(x, attn_mask, position_bias)
        elif self.style == "post-ln":
            return self.post_ln_forward(x, attn_mask, position_bias)
        else:
            raise ValueError("Unknown style: {}".format(self.style))


class AttentionBlockFactory:
    """Factory for AttentionBlock.
    
    Parameters:
        implementation (str): Implementation of the attention block. Default: "torch"
        kwargs: Additional arguments for the attention block.
        
    """
    def __init__(self, implementation="torch", **kwargs):
        self.attention_cls = AttentionBlock
        if implementation == "torch":
            self.attn = TorchMultiHeadAttention
        elif implementation == "own":
            self.attn = MultiHeadSelfAttention
        else:
            raise ValueError(
                "Unknown attention block implementation: {}".format(implementation))
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.attention_cls(*args, **{"attn":self.attn, **kwargs, **self.kwargs})
