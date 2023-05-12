from functools import partial
from typing import List
from torch.nn import Module, Linear, Softmax, Sequential, Identity
import torch
from einops import rearrange


class PredictorHead(Module):
    """A simple MLP head for ViT-based models.
    Intended for pooling across the frequency dimension and then applying a MLP on the pooled features.
    """
    def __init__(self,
                 patch_embed_dim,
                 num_classes,
                 classifier_hidden_dims: List[int],
                 *,
                 pooling_op="concat",
                 pooling_num_patches=None,
                 apply_softmax=False,
                 pool_axis=-3,
                 softmax_axis=-1):
        super().__init__()
        # num layers, output sizes layers, tag conditioning option
        assert softmax_axis == -1, "softmax axis must be -1 for now"  # TODO: make this more flexible
        if pooling_op == "concat":
            # b h w d -> b w h*d # pool_axis = -3
            assert pool_axis == -3, "pool_axis must be -3 for now"  # TODO: make this more flexible
            self.pooling = partial(rearrange, pattern="b h w d -> b w (h d)")
            assert pooling_num_patches is not None
            projection_input_dim = pooling_num_patches * patch_embed_dim
        elif pooling_op == "mean":  # mean along given axis, resulting in a mean patch of patch_size
            self.pooling = partial(torch.mean, dim=pool_axis)
            projection_input_dim = patch_embed_dim

        self.classifier = Sequential(*[Linear(i,o) for i,o in zip([projection_input_dim]+classifier_hidden_dims,
                                                                  classifier_hidden_dims+[num_classes])])
        self.softmax = Softmax(dim=softmax_axis) if apply_softmax else Identity()

    def forward(self, batch):
        """batch: Bx...xFxD
        we assume the frequency index for patches is given in  the second to last dimension here"""
        pooled = self.pooling(batch)
        logits = self.classifier(pooled)
        probs = self.softmax(logits)
        return probs


