from torch import nn
import torch
import math

from msm_mae.msm_mae.pos_embed import get_2d_sincos_pos_embed
from msm_mae.msm_mae.pos_embed import get_sinusoid_encoding_table


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim, use_class_token):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cls_token = use_class_token

    def forward(self, x):
        if self.use_cls_token:
            x = x + self.pos_embed[:, 1:x.shape[1]+1, :]
        else:
            x = x + self.pos_embed[:, :x.shape[1], :]
        return x

    def embed_cls(self, cls_token):
        assert self.use_cls_token
        return cls_token + self.pos_embed[:, :1, :]

class DummyPositionalEncoder(PositionalEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x

class FixedSizePositionalEncoderBase(PositionalEncoder):
    def __init__(self, total_patches, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_special_tokens = 1 if self.use_cls_token else 0
        self.total_patches = total_patches + num_special_tokens

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.total_patches, self.embed_dim), requires_grad=False
        )

class FixedSize1DPositionalEncoder(FixedSizePositionalEncoderBase):
    def __init__(self, sequence_length,  *args, **kwargs):
        super().__init__(total_patches=sequence_length, *args, **kwargs)
        self.sequence_length = sequence_length

class FixedSize2DPositionalEncoder(FixedSizePositionalEncoderBase):
    def __init__(self, grid_size, *args, **kwargs):
        super().__init__(total_patches=grid_size[0]*grid_size[1], *args, **kwargs)
        self.grid_size = grid_size

class RelativePositionalEncoder(PositionalEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SinCos2DPositionalEncoder(FixedSize2DPositionalEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.grid_size, cls_token=self.use_cls_token
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))


class SinCos1DPositionalEncoder(FixedSize1DPositionalEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pos_embed = get_sinusoid_encoding_table(
            self.sequence_length, self.pos_embed.shape[-1], cls_token=self.use_cls_token)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))


# taken from https://github.com/microsoft/unilm/blob/840b020ce5397c1c0dd9e62a2f4c66fba3afd934/beats/backbone.py#L33
class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class ConvolutionalPositionalEncoder(RelativePositionalEncoder):
    """Convolutional position encoder as used in WavLM and BEATs"""
    def __init__(self, kernel_size, groups, dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pos_conv = nn.Conv1d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        std = math.sqrt((4 * (1.0 - dropout)) /
                        (kernel_size * self.embed_dim))
        nn.init.normal_(pos_conv.weight, mean=0, std=std)
        nn.init.constant_(pos_conv.bias, 0)
        self.pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
        self.pos_embed = nn.Sequential(
            self.pos_conv, SamePad(kernel_size), nn.GELU())

    def forward(self, x):
        # convolution along patch dimension independently for groups sized chunks of the embedding dimension
        
        x_conv = self.pos_embed(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv
        return x
