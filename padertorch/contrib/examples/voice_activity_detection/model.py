import numpy as np
from padertorch.contrib.je.modules.conv import CNN1d, CNN2d
from padertorch.base import Model
import torch
import torch.nn as nn


class SAD_Classifier(Model):
    """This is an implementation of the neural network-based approach without an RNN from [1]_.

    References
    ----------
    .. [1] Heitkaemper, Jens, Joerg Schmalenstroeer, and Reinhold Haeb-Umbach. "Statistical and Neural Network Based Speech Activity Detection in Non-Stationary Acoustic Environments." arXiv preprint arXiv:2005.09913 (2020).
    """
    def __init__(self,
                 conv_layer: CNN2d,
                 temporal_layer: CNN1d,
                 pooling: nn.Module,
                 activation: nn.Module):
        super().__init__()
        self.conv_layer = conv_layer
        self.temporal_layer = temporal_layer
        self.pooling = pooling
        self.activation = activation

    def forward(self, batch: dict):
        x = batch['features']
        seq_len = x.shape[-1]
        x, _ = self.conv_layer(x)
        x = x.squeeze(2)
        x, _ = self.temporal_layer(x)
        x = self.pooling(x)
        x = x.squeeze(1)
        x = self.activation(x)
        output_len = x.shape[-1]
        scale_factor = np.ceil(seq_len/output_len)
        return np.repeat(x, scale_factor, axis=x.shape-1)[:, :seq_len]

    def review(self, inputs, outputs):
        activity = inputs['activity']
        bce = nn.BCELoss(reduction='sum')(outputs, activity)
        scalar_names = [
            'true_pos_{thres}',
            'false_pos_{thres}',
            'true_neg_{thres}',
            'false_neg_{thres}'
        ]
        results = {}
        for thres in [0.3, 0.5]:
            binarized_prediction = outputs > thres
            assert binarized_prediction.dtype == torch.bool, binarized_prediction.dtype
            boolean_activity = activity > 0
            assert boolean_activity.dtype == torch.bool, boolean_activity.dtype
            tp = torch.sum(binarized_prediction & boolean_activity).cpu().item()
            fp = torch.sum(binarized_prediction & ~boolean_activity).cpu().item()
            tn = torch.sum(~binarized_prediction & ~boolean_activity).cpu().item()
            fn = torch.sum(~binarized_prediction & boolean_activity).cpu().item()
            assert tp + fp + tn + fn == torch.numel(binarized_prediction), (tp + fp + tn + fn, torch.numel(binarized_prediction))
            keys = [name.format(thres=thres) for name in scalar_names]
            values = [tp, fp, tn, fn]
            results.update(zip(keys, values))

        summary = dict(
            loss=bce.sum(),
            scalars=results
        )
        return summary

    def modify_summary(self, summary):
        summary = super().modify_summary(summary)
        # compute precision, recall and fscore for each decision threshold
        scalar_names = [
            'true_pos_{thres}',
            'false_pos_{thres}',
            'true_neg_{thres}',
            'false_neg_{thres}'
        ]
        for thres in [0.3, 0.5]:
            if all([
                key in summary['scalars']
                for key in [name.format(thres=thres) for name in scalar_names]
            ]):
                tp, fp, tn, fn = [
                    np.sum(summary['scalars'][name.format(thres=thres)])
                    for name in scalar_names
                ]
                p = tp/(tp+fp)
                r = tp/(tp+fn)
                pfn = fn / (fn + tp)
                pfp = fp / (fp + tn)
                summary['scalars'][f'precision_{thres}'] = p
                summary['scalars'][f'recall_{thres}'] = r
                summary['scalars'][f'f1_{thres}'] = 2*(p*r)/(p+r)
                summary['scalars'][f'dcf_{thres}'] = 0.75 * pfn + 0.25 * pfp
                summary['scalars'][f'total_{thres}'] = tp+fp+tn+fn
        return summary
