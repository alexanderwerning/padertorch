from torch import nn
import torch
import math

from msm_mae.msm_mae.pos_embed import get_2d_sincos_pos_embed
from msm_mae.msm_mae.pos_embed import get_sinusoid_encoding_table
from einops import rearrange

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

    def load_state_dict(self, state_dict, strict):
        # weights are computed, skip loading
        missing, unexpected = [], []
        return missing, unexpected

class SinCos1DPositionalEncoder(FixedSize1DPositionalEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pos_embed = get_sinusoid_encoding_table(
            self.sequence_length, self.pos_embed.shape[-1], cls_token=self.use_cls_token)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))
        
class LearnedPositionalEncoder(FixedSize1DPositionalEncoder):
    def __init__(self, grid, init_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid = grid
        self.pos_embed.data.copy_(
            torch.randn(self.pos_embed.shape).float()
        )
        if init_path is not None:
            self.load_weights(init_path)


    def load_weights(self, path, transpose=True):
        state_dict = torch.load(path)
        self.pos_embed.data.copy_(state_dict['pos_embed'].transpose(-2,-3))
        # https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py
        # adaptation of ViT/DeiT positional embedding to spectrogram transformer

        # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
        new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.original_hw, self.original_hw)
        # cut (from middle) or interpolate the second dimension of the positional embedding
        if self.grid[1] <= self.original_hw:
            new_pos_embed = new_pos_embed[:, :, :, int(self.original_hw / 2) - int(self.grid[1] / 2): int(self.original_hw / 2) - int(self.grid[1] / 2) + self.grid[1]]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.original_hw, self.grid[1]), mode='bilinear')
        # cut (from middle) or interpolate the first dimension of the positional embedding
        if self.grid[0] <= self.original_hw:
            new_pos_embed = new_pos_embed[:, :, int(self.original_hw / 2) - int(self.grid[0] / 2): int(self.original_hw / 2) - int(self.grid[0] / 2) + self.grid[0], :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.grid[0], self.grid[1]), mode='bilinear')
        # flatten the positional embedding
        new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, self.num_patches).transpose(1,2)
        # concatenate the above positional embedding with the cls token and distillation token of the deit model.
        self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

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
        
        x_conv = self.pos_embed(x.transpose(-2, -1))
        x_conv = x_conv.transpose(-2, -1)
        x = x + x_conv
        return x

class Convolutional2DPositionalEncoder(RelativePositionalEncoder):
    """Extension of the convolutional position encoder to 2D inputs.
    
    """
    def __init__(self, grid_h, kernel_size=[9,128], groups=16, dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        if isinstance(kernel_size, (list, tuple)):
            padding = [kernel_size[0] // 2, kernel_size[1] // 2]
        self.grid_h = grid_h
        self.pos_conv = nn.Conv2d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        std = math.sqrt((4 * (1.0 - dropout)) /
                        (kernel_size[0]*kernel_size[1] * self.embed_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)
        # self.pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
        self.pos_embed = nn.Sequential(
            self.pos_conv, SamePad(kernel_size), nn.GELU())

    def forward(self, x):
        # convolution along patch dimension independently for groups sized chunks of the embedding dimension
        
        # batch_size, num_patches, embed_dim = x.shape
        B = x.shape[0]
        x = x.view(B, self.embed_dim, self.grid_h, -1)
        x_conv = self.pos_embed(x)
        x_conv = x_conv.view(B, -1, self.embed_dim)
        x = x + x_conv
        return x

class DisentangledPositionalEncoder(PositionalEncoder):
    """Disentangled 2D position encoder as used in PaSST.

    The position embedding is split into two parts: an embedding for the
    row position and an embedding for the column position. The row
    position embedding is shared across all columns, while the column
    position embedding is shared across all rows.

    Args:
    """

    def __init__(self, grid, h_enc='sincos1d', w_enc='learned', init=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid = grid
        # fixed time embedding
        if h_enc == 'learned':
            self.row_embed = nn.Parameter(
                torch.zeros(1, self.grid[1], self.embed_dim), requires_grad=True
            )
        elif h_enc == 'sincos1d':
           self.row_embed = SinCos1DPositionalEncoder(sequence_length=grid[1], *args, **kwargs)
        else:
            raise ValueError(f'Unknown h_enc: {h_enc}')
        # learnable frequency embedding
        if w_enc == 'learned':
            self.column_embed = nn.Parameter(
                torch.zeros(1, self.grid[0], self.embed_dim), requires_grad=True
            )
        else:
            raise ValueError(f'Unknown w_enc: {w_enc}')
        self.mode = h_enc, w_enc

        if self.use_cls_token:
            self.cls_token_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if init is not None:
            self.init(init)

    def embed_cls(self, cls_token):
        return cls_token + self.cls_token_embed

    def init(self, path):
        pt_file = torch.load(path)
        if 'model' in pt_file:
            loaded_weights = pt_file['model']
        else:
            loaded_weights = pt_file
            for n, p in self.named_parameters():
                if n in loaded_weights:
                    p.data = loaded_weights[n]
        if 'time_new_pos_embed' in loaded_weights and 'freq_new_pos_embed' in loaded_weights:
            # 'freq_new_pos_embed', 'time_new_pos_embed', 'new_pos_embed' for CLS and DIST tokens
            w_weights = loaded_weights['time_new_pos_embed']
            self.row_embed.data = rearrange(w_weights, '1 d 1 w -> 1 w d')
            h_weights = loaded_weights['freq_new_pos_embed']
            self.column_embed.data = rearrange(h_weights, '1 d h 1 -> 1 h d')
    
    def forward(self, x):
        # x: (B, P, D)
        if self.use_cls_token:
            cls_token = x[:,0, None]
            x = x[:, 1:]
        w = x.shape[1] // self.grid[0]
        if self.mode == ('learned', 'learned'):
            x = x + self.row_embed[:,:w].repeat(1, self.grid[0], 1) + self.column_embed.repeat_interleave(w, dim=1)
        elif self.mode == ('learned', 'sincos1d'):
            x = x + self.row_embed.pos_embed[:,:w].repeat(1, self.grid[0], 1) + self.column_embed.repeat_interleave(w, dim=1)
        return torch.cat([self.embed_cls(cls_token), x], dim=1)