from padertorch import Trainer
import numpy as np
import padertorch as pt
import torch

class AWTrainer(Trainer):
    """Trainer which allows to collect layer-wise grad norm stats"""
    def __init__(self, *args, clip_summary=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_summary = clip_summary

    def clip_grad(self, summary: dict):
        # TODO: report clipped and unclipped
        # TODO: allow clip=None but still report grad_norm

        summary.setdefault('scalars', {})
        summary.setdefault('histograms', {})

        def check(grad_norm):
            if not np.all(np.isfinite(pt.utils.to_numpy(grad_norm, detach=True))):
                # Write each interesting object to an individual file, because
                # not each object is serializable with `torch.save`.
                log_path_pattern = self.log_error_state({
                    'model': self.model,
                    'state_dict': self.state_dict(),
                    'optimizer_summary': summary,
                    'grad': {k: v.grad for k, v in self.model.named_parameters()},
                })
                raise RuntimeError(
                    f"The grad_norm ({grad_norm}) is not finite.\n"
                    f"See error states (model, example, model_out and review) in "
                    f"{log_path_pattern}."
                )

        if isinstance(self.optimizer, dict):
            for key, opti in self.optimizer.items():
                if self.clip_summary is not None:
                    summary = self.clip_summary(self.model, opti, summary, prefix=key)
                grad_norm = opti.clip_grad()
                check(grad_norm)

                summary['scalars'][f'{key}_grad_norm'] = grad_norm
                # underscore was necessary to obtain unique keys to prevent
                # tensorboard error
                summary['histograms'][
                    f'{key}_grad_norm_'] = torch.Tensor([grad_norm])
        else:
            if self.clip_summary is not None:
                summary = self.clip_summary(self.model, self.optimizer, summary)
            grad_norm = self.optimizer.clip_grad()
            check(grad_norm)
            summary['scalars'][f'grad_norm'] = grad_norm
            summary['histograms'][f'grad_norm_'] = \
                torch.Tensor([grad_norm])

        return summary
    
from torch.cuda.amp import autocast
from padertorch import Trainer
from torch.cuda.amp import GradScaler

class AutocastTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_scaler = GradScaler()


    def optimizer_step(self):
        if isinstance(self.optimizer, dict):
            for opti in self.optimizer.values():
                self.grad_scaler.unscale_(opti.optimizer)
        else:
            self.grad_scaler.unscale_(self.optimizer.optimizer)
        summary = self.clip_grad({})

        # Add learning rate to the summary
        if isinstance(self.optimizer, dict):
            for key, optim in self.optimizer.items():
                for i, param_group in enumerate(optim.optimizer.param_groups):
                    summary['scalars'][f'lr/{key}/param_group_{i}'] = param_group['lr']
        else:
            for i, param_group in enumerate(self.optimizer.optimizer.param_groups):
                summary['scalars'][f'lr/param_group_{i}'] = param_group['lr']

        # Do the actual optimization
        if isinstance(self.optimizer, dict):
            for opti in self.optimizer.values():
                self.grad_scaler.step(opti.optimizer)
        else:
            self.grad_scaler.step(self.optimizer.optimizer)

        self.grad_scaler.update()
        self.optimizer_zero_grad()
        return summary


    def step(self, *args, **kwargs):
        with autocast():
            loss, example, model_out, summary = super().step(*args, **kwargs)
            loss = self.grad_scaler.scale(loss)
            return loss, example, model_out, summary