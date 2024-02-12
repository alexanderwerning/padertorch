# adapted from timm.models.vision_transformer.PatchEmbed 
from itertools import repeat
import collections.abc
from torch import nn
import torch




class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding

    >>> patch_embed = PatchEmbed(patch_size=(16, 16), embed_dim=768)
    >>> input = torch.randn(4, 1, 80, 208)
    >>> x, grid = patch_embed(input)
    >>> x.shape
    torch.Size([4, 65, 768])
    """

    def __init__(
            self,
            patch_size=(16, 16),
            patch_overlap=(0, 0),
            embed_dim=768,
            output_dim=None,
            xavier_init=True,
            flatten_transpose=False,
            bias=True,
            init_path=None,
            learnable=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.stride = (patch_size[0] - patch_overlap[0], patch_size[1] - patch_overlap[1])
        assert self.stride[0] > 0 and self.stride[1] > 0, "Patch stride must be positive."

        self.proj = nn.Conv2d(
            1, embed_dim, kernel_size=patch_size, stride=self.stride, bias=bias)
        self.norm = nn.LayerNorm(embed_dim)

        if output_dim is not None:
            self.out_proj = nn.Linear(embed_dim, output_dim)
        else:
            self.out_proj = nn.Identity()

        self.flatten_transpose = flatten_transpose
        self.embed_dim = embed_dim
        if not learnable:
            self.proj.weight.requires_grad = False

        if xavier_init:
            # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
            w = self.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        if init_path is not None:
            self.image_net_init(init_path)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H - self.patch_size[0]) % self.stride[0] == 0, \
            f"Input height ({H}) minus patch size ({self.patch_size[0]}) must be divisible by patch stride ({self.stride[0]})."
        assert (W - self.patch_size[1]) % self.stride[1] == 0, \
            f"Input width ({W}) minus patch size ({self.patch_size[1]}) must be divisible by patch stride ({self.stride[1]})."
        grid = (H - self.patch_size[0]) // self.stride[0] + 1, (W - self.patch_size[1]) // self.stride[1] + 1
        # input_shape = H, W
        x = self.proj(x)
        if self.flatten_transpose:
            x = x.transpose(-1, -2)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        x = self.out_proj(x)
        return x, grid
    
    def image_net_init(self, image_net_path):
        pt_file = torch.load(image_net_path)
        if 'model' in pt_file:
            loaded_weights = pt_file['model']
        else:
            loaded_weights = pt_file
        if 'patch_embed.proj.weight' in loaded_weights:
            weight = loaded_weights['patch_embed.proj.weight']
            assert weight.shape[1] == 1 or weight.shape[1] == 3, \
                f"Expected 1 or 3 input channels, got {weight.shape[1]}."
            assert weight.shape[0] == self.embed_dim
            assert weight.shape[-2:] == tuple(self.patch_size)
            if weight.shape[1] == 3:
                weight = weight.mean(dim=1, keepdims=True)
            self.proj.weight.data = weight
        if 'patch_embedding.weight' in loaded_weights:
            weight = loaded_weights['patch_embedding.weight']
            assert weight.shape[1] == 1 or weight.shape[1] == 3, \
                f"Expected 1 or 3 input channels, got {weight.shape[1]}."
            assert weight.shape[0] == self.embed_dim
            assert weight.shape[-2:] == tuple(self.patch_size)
            if weight.shape[1] == 3:
                weight = weight.mean(dim=1, keepdims=True)
            self.proj.weight.data = weight
        if 'post_extract_proj.weight' in loaded_weights:
            assert self.out_proj is not None
            self.out_proj.weight.data = loaded_weights['post_extract_proj.weight']
            self.out_proj.bias.data = loaded_weights['post_extract_proj.bias']


def pad_spec(x, patch_size, patch_overlap):
    """pad spectrogram to be compatible with patch size and overlap
    
    Args:
        x: spectrogram, shape: (batch, 1, freq, time)
    
    Returns:
        x: spectrogram, shape: (batch, 1, freq, time)

    >>> x = torch.rand(2, 1, 257, 100)
    >>> pad_spec(x, (16, 16,), (0, 0)).shape
    torch.Size([2, 1, 257, 112])

    """
    _, pw = patch_size
    _, ow = patch_overlap
    w = x.shape[-1]
    w -= pw
    pad_len = 0
    if w < 0:
        pad_len = -w
    elif w % pw-ow != 0:
        pad_len = pw - (w % pw-ow)
    x = torch.nn.functional.pad(x, (0, pad_len))
    return x, pad_len