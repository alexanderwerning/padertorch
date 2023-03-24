from padertorch.train.optimizer import Optimizer
import torch
from torch import nn
from torch import optim

# TODO: decide how to replicate param_groups_weight_decay method here;
#  the model and named parameters are not available to this object
# a) separate set paramgroup method that is given the model object?
#  + assert that this function was called?


class AdamW(Optimizer):
    optimizer_cls = optim.AdamW
    parameters = None

    def __init__(
        self,
        gradient_clipping=1e10,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        super().__init__(
            gradient_clipping,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

    def param_groups_weight_decay(
        self, model: nn.Module, weight_decay=1e-5, no_weight_decay_list=()
    ):
        """Adapted from [1].

        [1] https://github.com/rwightman/pytorch-image-models/blob/main/timm/optim/optim_factory.py
        """
        no_weight_decay_list = set(no_weight_decay_list)
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if (
                param.ndim <= 1
                or name.endswith(".bias")
                or name in no_weight_decay_list
            ):
                no_decay.append(param)
            else:
                decay.append(param)

        self.parameters = [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay},
        ]

    def set_parameters(self, parameters):
        if self.parameters is not None:
            # "Parameter groups need to be set using param_groups_weight_decay"
            self.optimizer = self.optimizer_cls(
                self.parameters, **self.optimizer_kwargs
            )
        else:
            self.parameters = tuple(parameters)
            self.optimizer = self.optimizer_cls(
                self.parameters, **self.optimizer_kwargs
            )

    def clip_grad(self):
        self.check_if_set()
        grad_clips = self.gradient_clipping
        if isinstance(self.parameters, dict):
            return torch.nn.utils.clip_grad_norm_(
                [d["params"] for d in self.parameters], grad_clips
            )
        else:
            return torch.nn.utils.clip_grad_norm_(self.parameters, grad_clips)
