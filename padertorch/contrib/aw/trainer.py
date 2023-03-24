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