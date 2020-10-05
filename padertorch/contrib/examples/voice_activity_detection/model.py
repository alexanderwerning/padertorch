import numpy as np
from padertorch.contrib.je.modules.conv import CNN1d, CNN2d
from padertorch.base import Model
import torch
from torch.autograd import Variable
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
        seq_len = batch['seq_len']
        x, seq_len = self.conv_layer(x)
        x = x.squeeze(2)
        x, seq_len = self.temporal_layer(x)
        x = self.pooling(x)
        x = x.squeeze(1)
        x = self.activation(x)
        return x

    def review(self, inputs, outputs):
        activity = inputs['activity']
        bce = torch.nn.BCELoss(reduction='none')(outputs, activity)
        scalar_names = [
            'true_pos_{thres}',
            'false_pos_{thres}',
            'true_neg_{thres}',
            'false_neg_{thres}'
        ]
        results = {}
        for thres in [0.3, 0.5]:
            binarized_prediction = outputs > thres
            for key in [name.format(thres=thres) for name in scalar_names]:
                boolean_activity = activity > 0  # activity should already be binary?
                true = binarized_prediction == boolean_activity
                false = np.bitwise_not(true)
                assert np.sum(true + false) == binarized_prediction.size()
                tp = torch.sum(torch.bitwise_and(true, boolean_activity)).cpu().item()
                fp = torch.sum(torch.bitwise_and(false, boolean_activity)).cpu().item()
                tn = torch.sum(torch.bitwise_and(true, np.bitwise_not(boolean_activity))).cpu().item()
                fn = torch.sum(torch.bitwise_and(false, np.bitwise_not(boolean_activity))).cpu().item()
                results[key] = (tp, fp, tn, fn)

        summary = dict(
            loss=bce.mean(),
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
        return summary
