""" This module contains various hooks which perform actions during training.
"""
from collections import defaultdict
from enum import IntEnum
import json
import operator
import os

import numpy as np
import torch
from cached_property import cached_property
from tensorboardX import SummaryWriter

from padertorch.train.trigger import IntervalTrigger, EndTrigger, OrTrigger


__all__ = [
    'SummaryHook',
    'SimpleCheckpointHook',
    'ValidationHook',
    'CheckpointedValidationHook',
    'ProgressBarHook',
    'StopTrainingHook',
    'StopTraining',
]


class Priority(IntEnum):
    """
    Summary 50
    Print 40 NotImplemented
    ProgressBar(TQDM) 30 NotImplemented
    Validation 25
    Checkpoint 20
    End 10

    End has to be the last one
    Summary before Validation, clears timer information
    Print and ProgressBar may access Summary
    """
    END = 10
    DEFAULT = 15
    VALIDATION = 20
    CHECKPOINT = 25
    PROGRESS = 30
    PRINT = 40
    SUMMARY = 50


class BaseHook:

    def __init__(self, trigger=None):
        """
        :param trigger: Length of step between occurences or Trigger.
            It consists of an integer and either 'epoch' or 'iteration'
        """
        if trigger is not None:
            self.trigger = IntervalTrigger.new(trigger)

    @property
    def priority(self):
        return Priority.DEFAULT

    def pre_step(self, trainer: 'pt.Trainer'):
        """
        function is called before each iteration of the train iterator
        :param trainer:
        :return:
        """
        pass

    def post_step(self, trainer: 'pt.Trainer', example, model_output,
                  review):
        """
        function is called after each train step
        :param trainer:
        :param example:
        :param model_output:
        :param review:
        :return:
        """
        pass

    def close(self, trainer: 'pt.Trainer'):
        pass

    def set_last(self, iteration, epoch):
        self.trigger.set_last(iteration, epoch)


class SummaryHook(BaseHook):
    def __init__(self, trigger, validate=None,
                 summary_prefix='training'):
        super().__init__()

        if validate is None:
            super().__init__(trigger)
        else:
            super().__init__(OrTrigger(
                IntervalTrigger.new(trigger),
                IntervalTrigger.new(validate),
            ))
        self.reset_summary()
        self.summary_prefix = summary_prefix
        self.storage_dir = None

    @property
    def priority(self):
        return Priority.SUMMARY

    @cached_property
    def writer(self):
        return SummaryWriter(str(self.storage_dir),
                             filename_suffix=self.summary_prefix)

    @staticmethod
    def empty_summary_dict():
        return dict(
            losses=defaultdict(list),
            scalars=defaultdict(list),
            histograms=defaultdict(list),
            audios=dict(),
            images=dict()
        )

    def reset_summary(self):
        # Todo: add figures
        self.summary = self.empty_summary_dict()

    def update_summary(self, review):
        for key, loss in review.get('losses', dict()).items():
            self.summary['losses'][key].append(loss.item())
        for key, scalar in review.get('scalars', dict()).items():
            self.summary['scalars'][key].append(
                scalar.item() if torch.is_tensor(scalar) else scalar)
        for key, histogram in review.get('histograms', dict()).items():
            self.summary['histograms'][key] = np.concatenate(
                [self.summary['histograms'].get(key, np.zeros(0)),
                 histogram.clone().cpu().data.numpy().flatten()]
            )[-10000:]  # do not hold more than 10K values in memory
        for key, audio in review.get('audios', dict()).items():
            self.summary['audios'][key] = audio  # snapshot
        for key, image in review.get('images', dict()).items():
            self.summary['images'][key] = image  # snapshot

    def dump_summary(self, trainer: 'pt.Trainer'):
        iteration = trainer.iteration
        timer = trainer.timer
        prefix = self.summary_prefix
        for key, loss in self.summary['losses'].items():
            self.writer.add_scalar(
                f'{prefix}/{key}', np.mean(loss), iteration)
        for key, scalar in self.summary['scalars'].items():
            self.writer.add_scalar(
                f'{prefix}/{key}', np.mean(scalar), iteration)
        for key, scalar in timer.as_dict.items():
            if key in ['time_per_data_loading', 'time_per_train_step']:
                if 'time_per_step' in timer.as_dict.keys():
                    time_per_step = timer.as_dict['time_per_step']
                    if len(time_per_step) != len(scalar):
                        print(
                            'Warning: padertorch.Trainer timing bug.'
                            f'len(time_per_step) == {len(time_per_step)} '
                            f'!= len(scalar) == {len(scalar)}'
                        )
                    scalar = (
                        scalar.sum() / time_per_step.sum()
                    )
                    if key == 'time_per_data_loading':
                        key = 'time_rel_data_loading'
                    elif key == 'time_per_train_step':
                        key = 'time_rel_train_step'
                else:
                    # Something went wrong, most likely an exception.
                    pass
            self.writer.add_scalar(
                f'{prefix}/{key}', scalar.mean(), iteration)
        for key, histogram in self.summary['histograms'].items():
            self.writer.add_histogram(
                f'{prefix}/{key}', np.array(histogram), iteration
            )
        for key, audio in self.summary['audios'].items():
            if isinstance(audio, (tuple, list)):
                assert len(audio) == 2, (len(audio), audio)
                self.writer.add_audio(
                    f'{prefix}/{key}', audio[0],
                    iteration, sample_rate=audio[1]
                )
            else:
                self.writer.add_audio(
                    f'{prefix}/{key}', audio,
                    iteration, sample_rate=16000
                )
        for key, image in self.summary['images'].items():
            self.writer.add_image(f'{prefix}/{key}', image, iteration)
        self.reset_summary()
        trainer.reset_timer()

    def pre_step(self, trainer: 'pt.Trainer'):
        if(self.trigger(iteration=trainer.iteration, epoch=trainer.epoch)
           or trainer.iteration == 1):
            self.dump_summary(trainer)

    def post_step(self, trainer: 'pt.Trainer', example, model_out, review):
        if self.storage_dir is None:
            self.storage_dir = trainer.storage_dir
        else:
            assert self.storage_dir == trainer.storage_dir
        self.update_summary(review)

    def close(self, trainer: 'pt.Trainer'):
        self.dump_summary(trainer)


class SimpleCheckpointHook(BaseHook):
    """ Can be used to keep all checkpoints, e.g. for continuous evaluation
            (latest_only = False) or to only store the most recent checkpoint
            (latest_only = True).
            Cannot be used together with a CheckpointedValidationHook
    """
    def __init__(self, trigger, latest_only=False):
        super().__init__(trigger)
        self.latest_only = latest_only
        self.last_checkpoint_path = None

    @property
    def priority(self):
        return Priority.CHECKPOINT

    def pre_step(self, trainer: 'pt.Trainer'):
        checkpoint_path = trainer.default_checkpoint_path()
        trainer.save_checkpoint(checkpoint_path)
        if self.latest_only and os.path.exists(self.last_checkpoint_path):
            os.remove(self.last_checkpoint_path)
        self.last_checkpoint_path = checkpoint_path


class ValidationHook(SummaryHook):
    def __init__(self, trigger, iterator):
        super().__init__(trigger, summary_prefix='validation')
        self.iterator = iterator

    @property
    def priority(self):
        return Priority.VALIDATION

    def pre_step(self, trainer: 'pt.Trainer'):
        assert all([len(value) == 0 for value in self.summary.values()])
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch):
            assert len(trainer.timer.timings) == 0, trainer.timer
            print('Starting Validation')
            for model_out, review in trainer.validate(self.iterator):
                self.update_summary(review)
            self.dump_summary(trainer)
            assert len(trainer.timer.timings) == 0, trainer.timer
            print('Finished Validation')

    def post_step(self, trainer: 'pt.Trainer', example, model_out, review):
        pass

    def close(self, trainer: 'pt.Trainer'):
        pass


class _Metric:
    """ Bookkeeping of metrics (comparison, best value, checkpoint path,
        symlink) needed for CheckpointedValidationHook.
    """
    def __init__(self, metric_key, criterion):
        self.key = metric_key
        self.criterion = criterion
        self.path = None
        self.symlink_name = None

        if criterion == 'min':
            self._is_better_fn = operator.lt
            self.value = float('inf')
        elif criterion == 'max':
            self._is_better_fn = operator.gt
            self.value = -float('inf')
        else:
            raise ValueError("Comparison criterion must be either"
                             " 'min' or 'max'!")

    def is_better(self, value):
        """ Decides whether current metric value is better than best
            previous one. Has to work for cost and gain objectives
            => See init for details.
        """
        return self._is_better_fn(value, self.value)

    def update(self, value, checkpoint_path):
        """ Update to best metric value, corresponding checkpoint path
            and set symlink to checkpoint.
        """
        self.value = value
        self.path = checkpoint_path
        self.symlink_name = (os.path.dirname(checkpoint_path) + os.path.sep
                             + 'ckpt_best_{metric_key}')
        os.symlink(checkpoint_path, self.symlink_name)

    def to_json(self):
        """ Dump metric state information into dictionary. """
        return dict(key=self.key, value=self.value,
                    criterion=self.criterion, path=self.path,
                    symlink_name=self.symlink_name)


class CheckpointedValidationHook(ValidationHook):
    """ Performs model validation and keeps checkpoints for model states that
        perform best on a given set of metrics.
        Cannot be used together with a ValidationHook
        or a SimpleCheckpointHook.
    """
    def __init__(self, trigger, iterator, metrics=None):
        super().__init__(trigger, iterator)
        assert isinstance(metrics, dict) and metrics,  \
            'The metrics dict must not be empty!'
        self.metrics = self._convert_metrics_to_internal_layout(metrics)
        self.best_validated_checkpoints = {}
        self.latest_checkpoint_path = None

    def dump_summary(self, trainer: 'pt.Trainer'):
        """ This class needs to overload the dump_summary - even if the naming
            is suboptimal - because the ValidationHook class produces the
            necessary metrics in its pre_step but immediately calls
            dump_summary. However, the implementation in SummaryHook clears
            the summary content.
        """
        self._save_latest_checkpoint(trainer)
        self._update_validated_checkpoints(trainer)
        super().dump_summary(trainer)

    def close(self, trainer: 'pt.Trainer'):
        self._save_latest_checkpoint(trainer)

    @classmethod
    def _convert_metrics_to_internal_layout(cls, metrics):
        return {metric_key: _Metric(metric_key, criterion)
                for metric_key, criterion in metrics.items()}

    def _save_latest_checkpoint(self, trainer):
        """ Unconditionally save a checkpoint for the current model.
            This is needed for resuming training.
        """
        checkpoint_path = trainer.default_checkpoint_path()
        trainer.save_checkpoint(checkpoint_path)
        if self.latest_checkpoint_path is not None:
            os.remove(self.latest_checkpoint_path)
        self.latest_checkpoint_path = checkpoint_path

    def _update_validated_checkpoints(self, trainer: 'pt.Trainer'):
        """ Save a checkpoint if the current model improves one or multiple
            validation metrics, dump the metric information to a json file
            and remove old checkpoints.
        """
        for metric_key, summary_value in self._relevant_summary().items():
            if self.metrics[metric_key].is_better(summary_value):
                self._update_checkpoint(trainer, metric_key, summary_value)
        self._dump_metrics_to_json()
        self._cleanup_stale_checkpoints()

    def _relevant_summary(self):
        # Check if all desired metrics are valid (i.e. appear in summary)
        invalid_metrics = (set(self.metrics.keys()) -
                           set(self.summary['scalars'].keys()))
        if invalid_metrics:
            raise ValueError(f"The metrics {invalid_metrics} do not exist in "
                             f"summary['scalars'].")
        # return all desired metrics from summary
        return {metric_key: summary_value
                for metric_key, summary_value in self.summary['scalars']
                if metric_key in self.metrics}

    def _update_checkpoint(self, trainer, metric_key, summary_value):
        """ Is called if a metric was improved with the current model state.
            Stores a checkpoint for the current state if not yet done
            and updates the corresponding metric object.
        """
        checkpoint_path = f'{trainer.default_checkpoint_path()}_best'
        if checkpoint_path not in self.best_validated_checkpoints:
            trainer.save_checkpoint(checkpoint_path)
        self.metrics[metric_key].update(summary_value, checkpoint_path)

    def _dump_metrics_to_json(self):
        """ Store the state information of the metrics objects to a json.
        """
        assert all(metric_key == metric.name
                   for metric_key, metric in self.metrics.items()), \
            'Some metric keys do not match their names!'
        json_path = (os.path.dirname(self.latest_checkpoint_path) +
                     os.path.sep + 'metrics.json')
        content = {metric_key: metric.to_json()
                   for metric_key, metric in self.metrics.items()}
        with open(json_path, 'w') as json_file:
            json.dump(content, json_file)

    def _cleanup_stale_checkpoints(self):
        """ Remove all checkpoints that became stale (i.e. have associated
            metric where they perform best anymore).
        """
        best_checkpoint_paths = {metric.path
                                 for metric in self.metrics.values()
                                 if metric.path is not None}
        for checkpoint_path in self.best_validated_checkpoints:
            if checkpoint_path not in best_checkpoint_paths:
                os.remove(checkpoint_path)
        self.best_validated_checkpoints = best_checkpoint_paths


class ProgressBarHook(BaseHook):
    """ Adds a progress bar to the console output. """
    def __init__(self, max_trigger, max_it_len=None, update_intervall=1):
        from progressbar import bar, UnknownLength
        """
        :param max_trigger: has to be defined if max_trigger unit is session
            integer with the length of the iterator
        :param max_it_len (int): length of iterator, only used if max_trigger
            uses unit epoch
        :param update_interval (int): Number of iterations to skip printing the
            progress bar.
        :param bar_length (int): Length of the progress bar in characters.
        :param disable: bool use to disable the entire progressbar wrapper
        """
        super().__init__((update_intervall, 'iteration'))
        self.ep_trigger = IntervalTrigger(1, 'epoch')
        self.update_intervall = update_intervall
        if isinstance(max_trigger, EndTrigger):
            length, unit = max_trigger.period, max_trigger.unit
        elif isinstance(max_trigger, (tuple, list)):
            length, unit = max_trigger
        else:
            raise ValueError(f'max_trigger is expected to be either a trigger'
                             f'or a list or tuple, but is {type(max_trigger)},'
                             f'{max_trigger}')
        if unit == 'iteration':
            max_iteration = length
        elif unit == 'epoch':
            if max_it_len is not None:
                max_iteration = length * max_it_len
            else:
                max_iteration = UnknownLength
        else:
            raise ValueError(f'unit {unit} is unknown,'
                             f' choose iteration or epoch')
        # self.pbar = tqdm(desc=f'epochs: {0}', total=max_iteration,
        #                  file=sys.stdout, ncols=ncols,
        #                  dynamic_ncols=dynamic_ncols)
        self.pbar = bar.ProgressBar(
            prefix=f'epochs: {0}',
            max_value=max_iteration,
            redirect_stderr=True,
            redirect_stdout=True
        )

    @property
    def priority(self):
        return Priority.PROGRESS

    def set_last(self, iteration, epoch):
        super().set_last(iteration, epoch)
        self.ep_trigger.set_last(iteration, epoch)
        self.pbar.value = iteration

    def post_step(self, trainer: 'pt.Trainer', example,
                  model_output, review):
        from progressbar import UnknownLength
        iteration = trainer.iteration
        epoch = trainer.epoch
        if self.trigger(iteration, epoch):
            self.pbar.update(self.update_intervall)
            if len(review['losses']) == 1:
                self.pbar.prefix = f'epochs: {epoch}, loss: ' \
                                 f'{list(review["losses"].values())[0]}'
        if self.ep_trigger(iteration, epoch) and self.pbar.max_value is UnknownLength:
            self.pbar.max_value = iteration * epoch


    def close(self, trainer: 'pt.Trainer'):
        self.pbar.finish()


class StopTrainingHook(BaseHook):
    """ Raises a StopTraining exception if triggered. """
    def __init__(self, trigger):
        super().__init__()
        self.trigger = EndTrigger.new(trigger)

    @property
    def priority(self):
        return Priority.END

    def pre_step(self, trainer):
        if self.trigger(trainer.iteration, trainer.epoch):
            print(f'Training ended after {trainer.epoch} epochs and'
                  f' {trainer.iteration} iterations')
            raise StopTraining


class StopTraining(Exception):
    """ Rationale: Raised as signal to stop the training
        (e.g. when predefined number of iterations are completed.)
    """
    pass
