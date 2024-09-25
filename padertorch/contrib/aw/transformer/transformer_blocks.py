"""Implementation of Transformer blocks, Self-Attention, Multi-head attention, gated relative position bias, deep norm?"""

import torch
import torch.nn as nn
from typing import Optional
import math
from einops import einsum

from padertorch.modules.fully_connected import fully_connected_stack

from padertorch.contrib.aw.transformer.attention_bias import (
    RelativePositionalBiasFactory,
)


class TorchMultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        add_bias_kv=False,
        dropout=0.0,
        attn_dropout=0.0,
        proj_dropout=0.0,
        use_k_bias=False,
        rel_pos_bias_factory=False,
    ):
        super().__init__()
        self._attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=add_bias_kv,
            batch_first=True,
        )
        self.in_proj_weight = self._attn.in_proj_weight
        self.out_proj = self._attn.out_proj
        self.scale = (dim // num_heads) ** -0.5
        assert (
            rel_pos_bias_factory is False or rel_pos_bias_factory is None
        ), "rel_pos_bias must be False for TorchMultiHeadAttention"

    def forward(self, x, attn_mask=None, position_bias=None, return_weights=False):
        assert (
            position_bias is None
        ), "position_bias must be None for StandardMultiHeadAttention"
        x, attn_weights = self._attn(self.scale * x, x, x, attn_mask=attn_mask)
        if return_weights:
            return x, attn_weights
        else:
            return x, None


# TODO: improve architecture, avoid passing position_bias around
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        dropout=0.0,
        attn_dropout=0.0,
        proj_dropout=0.0,
        rel_pos_bias_factory: Optional[RelativePositionalBiasFactory] = None,
        use_k_bias=True,
    ):
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
            self.rel_pos_bias = rel_pos_bias_factory(
                num_heads=num_heads, q_head_dim=head_dim
            )
        else:
            self.rel_pos_bias = None

        # compatibility with TorchMultiHeadAttention
        self.in_proj_weight = self.qkv.weight
        self.out_proj = self.proj

    def forward(self, x, attn_mask=None, position_bias=None, return_weights=False):
        B, N, C = x.shape

        # TODO: solve this mess
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
        qkv = nn.functional.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)

        # reshape to 3, B, num_heads, N, C // num_heads (head dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)

        q = q * self.scale  # TODO: this is not in-place, unlike before

        # training stabilization for mixed precision training
        # Noted in WavLM: D. Stabilization of Training
        # exploits translation invariance of softmax
        alpha = 32
        q *= 1 / alpha

        # attn shape: B, num_heads, N, N
        # attn = q @ k.transpose(-2, -1)
        attn = einsum(q, k, "b h l d, b h m d -> b h l m")

        # stabilize attention, see above
        attn = (attn - attn.max(dim=-1, keepdim=True)[0]) * alpha

        if attn_mask is not None:
            # attn_mask.shape == (N, N)
            attn += attn_mask

        if self.rel_pos_bias is not None:
            attn_bias, position_bias = self.rel_pos_bias(
                q * alpha / self.scale, position_bias
            )
            attn += attn_bias.view(B, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = einsum(attn, v, "b h l m, b h m d -> b l h d").reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        if not return_weights:
            attn = None
        if self.rel_pos_bias:
            return x, attn, position_bias
        else:
            return x, attn


class AttentionBlock(nn.Module):
    """ "Transformer Block.

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
        rel_pos_bias_factory: Optional[RelativePositionalBiasFactory] = None,
        act_layer="gelu",
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

        self.use_rel_pos_bias = bool(rel_pos_bias_factory)

        self.norm2 = norm_layer(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = fully_connected_stack(
            input_size=dim,
            hidden_size=[mlp_hidden_dim],
            output_size=dim,
            activation=act_layer,
            dropout=dropout,
        )

        self.residual_scale = residual_scale
        self.style = style

    def pre_ln_forward(
        self, x, attn_mask=None, position_bias=None, return_weights=False
    ):
        if self.use_rel_pos_bias:
            x_attn, attn_weights, position_bias = self.attn(
                self.norm1(x), attn_mask, position_bias, return_weights=return_weights
            )
        else:
            x_attn, attn_weights = self.attn(
                self.norm1(x), attn_mask, return_weights=return_weights
            )

        x = self.residual_scale * x + x_attn
        x = self.residual_scale * x + self.mlp(self.norm2(x))

        if self.use_rel_pos_bias:
            return x, attn_weights, position_bias
        else:
            return x, attn_weights

    def post_ln_forward(
        self, x, attn_mask=None, position_bias=None, return_weights=False
    ):
        if self.use_rel_pos_bias:
            x_attn, attn_weights, position_bias = self.attn(
                x, attn_mask, position_bias, return_weights=return_weights
            )
        else:
            x_attn, attn_weights = self.attn(
                x, attn_mask, return_weights=return_weights
            )

        x = self.residual_scale * x + x_attn
        x = self.norm1(x)
        x = self.residual_scale * x + self.mlp(x)
        x = self.norm2(x)

        if self.use_rel_pos_bias:
            return x, attn_weights, position_bias
        else:
            return x, attn_weights

    def forward(self, x, attn_mask=None, position_bias=None, return_weights=False):
        if self.style == "pre-ln":
            return self.pre_ln_forward(x, attn_mask, position_bias, return_weights)
        elif self.style == "post-ln":
            return self.post_ln_forward(x, attn_mask, position_bias, return_weights)
        else:
            raise ValueError("Unknown style: {}".format(self.style))


class ConformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.0,
        attn_dropout=0.0,
        expansion_factor=2,
        kernel_size=32,
        rel_pos_bias_factory: Optional[RelativePositionalBiasFactory] = None,
        act_layer="gelu",
        norm_layer=nn.LayerNorm,
        attn=TorchMultiHeadAttention,
    ):
        super().__init__()
        self.feed_forward1 = nn.Sequential(
            norm_layer(dim, eps=1e-6),
            nn.Linear(dim, mlp_ratio * dim),
            nn.SiLU(),  # Swish
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim),
            nn.Dropout(dropout),
        )

        self.feed_forward2 = nn.Sequential(
            norm_layer(dim, eps=1e-6),
            nn.Linear(dim, mlp_ratio * dim),
            nn.SiLU(),  # Swish
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim),
            nn.Dropout(dropout),
        )

        self.convolution_module = nn.Sequential(
            norm_layer(dim, eps=1e-6),
            nn.Conv1d(
                dim, expansion_factor * dim, kernel_size=1
            ),  # 2 x expansion factor
            nn.GLU(),
            nn.Conv1d(
                expansion_factor * dim,
                expansion_factor * dim,
                kernel_size=kernel_size,
                padding=1,
                groups=dim,
            ),  # depthwise convolution
            nn.BatchNorm1d(dim),
            nn.SiLU(),  # Swish
            nn.Conv1d(
                expansion_factor * dim, dim, kernel_size=1
            ),  # pointwise convolution
            nn.Dropout(dropout),
        )
        self.attn_norm = norm_layer(dim, eps=1e-6)
        self.final_norm = norm_layer(dim, eps=1e-6)
        self.attn = attn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout=dropout,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            rel_pos_bias_factory=rel_pos_bias_factory,
        )

        assert bool(rel_pos_bias_factory)

    def forward(self, x, attn_mask=None, position_bias=None, return_weights=False):
        x = x + 0.5 * self.feed_forward1(x)
        x_, attn_weights = self.attn(
            self.attn_norm(x), attn_mask, position_bias, return_weights=return_weights
        )
        x = x + self.attn_dropout(x_)
        x = x + self.convolution_module(x)
        x = x + 0.5 * self.feed_forward2(x)
        x = self.final_norm(x)
        return x, attn_weights, position_bias


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
        # elif implementation == "torch-cross":
        #     self.attn = TorchCrossAttention
        # elif implementation == "own-cross":
        #     self.attn = MultiHeadCrossAttention
        elif implementation == "conformer":
            self.attn = ConformerAttention
        else:
            raise ValueError(
                "Unknown attention block implementation: {}".format(implementation)
            )
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.attention_cls(*args, **{"attn": self.attn, **kwargs, **self.kwargs})
