import math
from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn

from padertorch.contrib.aw.transformer_blocks import RelativePositionalBiasFactory

from padertorch.contrib.aw.transformer_blocks import AttentionBlockFactory

# these functions were adapted from the pytorch transformer tutorial (language modeling)
# TODO: integrate into transformer, provide in forward, or rename to ViTEncoder?


def generate_forward_mask(sz: int, grid_size: Tuple[int]) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag.
    Used for forward model, which cannot see future tokens."""
    h, w = grid_size
    assert h*w == sz
    return torch.tril(torch.ones(h, h, w, w) * float('-inf'), diagonal=-1).moveaxis(-1, 1).reshape(sz, sz)


def generate_backward_mask(sz: int, grid_size: Tuple[int]) -> torch.Tensor:
    """Generates an lower-triangular matrix of -inf, with zeros on diag.
    Used for backward model, which cannot see past tokens."""
    h, w = grid_size
    assert h*w == sz
    return torch.triu(torch.ones(h, h, w, w) * float('-inf'), diagonal=1).moveaxis(-1, 1).reshape(sz, sz)

# TODO: work on forward backward parameters
class TransformerEncoder(nn.Module):
    """Transformer encoder module with optional time masking and relative positional bias.
    The masking is designed for a ViT-like encoder, where the input is a 2D grid of tokens.

    >>> transformer = TransformerEncoder(forward=False)
    >>> np.random.seed(3)
    >>> grid = (5, 13)
    >>> num_patches = 5*13 # 65
    >>> x = torch.randn(4, num_patches, 768)
    >>> outputs = transformer(x=x, grid_size=grid)
    >>> outputs[0].shape
    torch.Size([4, 65, 768])

    """
    @classmethod
    def finalize_dogmatic_config(cls, config):
        config["block_factory"] = {
            "factory": "padertorch.contrib.aw.transformer_blocks.AttentionBlockFactory",
        }
        config["norm_layer"] = {
            "factory": "padertorch.configurable.import_class",
            "name": "torch.nn.LayerNorm",
        }

    def __init__(
            self, *,
            embed_dim=768,
            depth=12,
            num_heads=12,
            output_dim=None,
            input_dim=None,
            mlp_ratio=4.0,
            block_factory=AttentionBlockFactory(),
            norm_layer=nn.LayerNorm,
            dropout=0,
            attn_dropout=0,
            layer_dropout=0,
            forward=True,
            backward=True,
            rel_pos_bias_factory: Optional[RelativePositionalBiasFactory] = None,
            init_mode="default",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim if output_dim is not None else embed_dim
        self.output_dim = output_dim if output_dim is not None else embed_dim
        self.layer_dropout = layer_dropout
        self.use_rel_pos_bias = rel_pos_bias_factory is not None
        self.blocks = nn.ModuleList(
            [
                block_factory(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    norm_layer=norm_layer,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    rel_pos_bias_factory=rel_pos_bias_factory
                )
                for _ in range(depth)
            ]
        )
        if input_dim is None:
            self.in_proj = None
        else:
            self.in_proj = nn.Linear(input_dim, embed_dim)

        if output_dim is None:
            self.out_proj = None
        else:
            self.out_proj = nn.Linear(embed_dim, output_dim)
        assert forward or backward
        self.forward_attn = forward
        self.backward_attn = backward
        self.layer_norm = norm_layer(embed_dim)
        self.apply_layernorm_before = self.blocks[0].style == "post-ln"
        if not self.apply_layernorm_before:
            assert self.blocks[0].style == "pre-ln"
        self.reset_parameters(init_mode=init_mode)

    # TODO: attention mask for sequence length
    def forward(self, x, grid_size=None):
        seq_len = x.shape[-2]
        if self.forward_attn and not self.backward_attn:
            assert grid_size is not None, "grid_size must be set for forward attention"
            src_mask = generate_forward_mask(
                seq_len, grid_size=grid_size).to(x.device)
        elif not self.forward_attn and self.backward_attn:
            assert grid_size is not None, "grid_size must be set for backward attention"
            src_mask = generate_backward_mask(
                seq_len, grid_size=grid_size).to(x.device)
        else:
            src_mask = None

        if self.in_proj is not None:
            x = self.in_proj(x)
        
        if self.apply_layernorm_before:
            x = self.layer_norm(x)

        position_bias = None
        for blk in self.blocks:
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layer_dropout):
                if self.use_rel_pos_bias:
                    x, position_bias = blk(
                        x, attn_mask=src_mask, position_bias=position_bias)
                else:
                    x = blk(x, attn_mask=src_mask)

        if not self.apply_layernorm_before:
            x = self.layer_norm(x)

        if self.out_proj is not None:
            x = self.out_proj(x)

        return x, seq_len

    def reset_parameters(self, init_mode="default"):
        if init_mode == "deep_norm":
            # block class should be post_LN
            assert self.blocks[0].style == "post-ln"
            deep_norm_alpha =  math.pow(2 * len(self.blocks), 1 / 4)
            deep_norm_beta = math.pow(8 * len(self.blocks), -1 / 4)
            for blk in self.blocks:
                blk.residual_scale = deep_norm_alpha
                attn = blk.attn
                mlp = blk.mlp
                # in_proj = concat(k_proj, q_proj, v_proj)
                nn.init.xavier_normal_(attn.in_proj_weight[:self.embed_dim], gain=1)  # k_proj.weight
                nn.init.xavier_normal_(attn.in_proj_weight[self.embed_dim:2*self.embed_dim], gain=deep_norm_beta) # v_proj.weight
                nn.init.xavier_normal_(attn.in_proj_weight[2*self.embed_dim:], gain=1) # q_proj.weight
                nn.init.xavier_normal_(attn.out_proj.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(mlp.linear_0.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(mlp.linear_1.weight, gain=deep_norm_beta)
        elif init_mode == "default":
            # Taken from https://github.com/microsoft/unilm/blob/master/beats/backbone.py
            
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            for blk in self.blocks:
                module = blk.attn
                nn.init.xavier_uniform_(module.in_proj_weight, gain=1 / math.sqrt(2))
                if module.in_proj_bias is not None:
                    # approximately xavier_normal_, but adapted for 1D
                    nn.init.normal_(module.in_proj_bias, 0.0, math.sqrt(2/module.embed_dim))

                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0.0)
                # if self.has_relative_attention_bias:
                #     nn.init.xavier_normal_(self.relative_attention_bias.weight)
        elif init_mode == "xlm":
            # noted in XLM paper (few details given)
            for i, blk in enumerate(self.blocks):
                beta = 1/math.sqrt(2*(i+1))  # enumerating layers starting with 1
                attn = blk.attn
                mlp = blk.mlp
                # in_proj = concat(k_proj, q_proj, v_proj)
                nn.init.xavier_normal_(attn.in_proj_weight[:self.embed_dim], gain=1)  # k_proj.weight
                nn.init.xavier_normal_(attn.in_proj_weight[self.embed_dim:2*self.embed_dim], gain=beta) # v_proj.weight
                nn.init.xavier_normal_(attn.in_proj_weight[2*self.embed_dim:], gain=1) # q_proj.weight
                nn.init.xavier_normal_(attn.out_proj.weight, gain=beta)
                nn.init.xavier_normal_(mlp.linear_0.weight, gain=beta)
                nn.init.xavier_normal_(mlp.linear_1.weight, gain=beta)


