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
            image_net_init_path=None,
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
        
        if image_net_init_path is not None:
            self.image_net_init(image_net_init_path)

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
        loaded_weights = torch.load(image_net_path)['model']['patch_embed.proj.weight']
        assert loaded_weights.shape[1] == 1 or loaded_weights.shape[1] == 3, \
            f"Expected 1 or 3 input channels, got {loaded_weights.shape[1]}."
        assert loaded_weights.shape[0] == self.embed_dim
        assert loaded_weights.shape[-2:] == tuple(self.patch_size)
        if loaded_weights.shape[1] == 3:
            loaded_weights = loaded_weights.mean(dim=1, keepdims=True)
        self.proj.weight.data = loaded_weights
