"""
    This module contains the Trainer class which can be used to train
    configurable padertorch models.
"""
import contextlib
import itertools
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import functools
import collections

import numpy as np
import torch
import torch.nn
import tensorboardX

from paderbox.utils.nested import deflatten
import padertorch as pt
from padertorch.configurable import Configurable
from padertorch.train.optimizer import Optimizer, Adam
from padertorch.train.runtime_tests import test_run
from padertorch.train.hooks import *
from padertorch.train.trigger import AnyTrigger

__all__ = [
    'Trainer',
    'InteractiveTrainer',
]


class Trainer(Configurable):

    @classmethod
    def finalize_dogmatic_config(cls, config):
        if 'optimizer' not in config.keys():
            config['optimizer'] = {'factory': Adam}

    def __init__(
            self,
            model,
            storage_dir,
            optimizer,
            loss_weights=None,
            summary_trigger=(1, 'epoch'),
            checkpoint_trigger=(1, 'epoch'),
            keep_all_checkpoints=False,
            max_trigger=(1, 'epoch'),
            virtual_minibatch_size=1,
    ):
        """

        Args:
            model: a `padertorch.base.Model` object
            storage_dir: The structure of produced storage_dir is:
                .
                ├── checkpoints
                │   ├── ckpt_7122.pth
                │   ├── ckpt_14244.pth
                │   ├── ckpt_best_loss.pth -> ckpt_7122.pth
                │   ├── ckpt_latest.pth -> ckpt_14244.pth
                │   └── ckpt_state.json
                ├── events.out.tfevents.1548851867.ntsim5
            optimizer: a `padertorch.train.optimizer.Optimizer` object
                or dict of Optimizers
            loss_weights: dict of weights for model with multiple losses
            summary_trigger: `pytorch.train.trigger.IntervalTrigger` object
                or tuple describing the interval when summaries
                are written to event files.
                See padertorch.train.hooks.SummaryHook for a description of
                what a summary is.
            checkpoint_trigger: `padertorch.train.trigger.IntervalTrigger`
                object or tuple describing the interval when checkpoints
                are saved.
                See padertorch.train.hooks.CheckpointedValidationHook for a
                description of what happens on a checkpoint.
            keep_all_checkpoints: flag if False only latest and best
                checkpoints are kept otherwise all checkpoints are kept
            max_trigger: `padertorch.train.trigger.EndTrigger` object
                or tuple describing the endpoint of the training
            virtual_minibatch_size: Runs the optimisation in
                virtual_minibatch_size steps. By default run it after each
                review call.
                The advantage of a virtual_minibatch_size over addressing a
                minibatch dimension in forward and review is a lower memory
                footprint on cost of cpu time.
                Note: The gradients are accumulated and not averaged.
                Note: The virtual_minibatch_size is fixed and can contain data
                    from two epochs.


        Usage:

            # For test_run we recommend to do it without prefetch
            trainer = Trainer(...)  # or: Trainer.from_config(...)
            trainer.test_run(tr_ds, val_ds)
            trainer.train(tr_ds.prefetch(4, 8), val_ds_with.prefetch(4, 8))

        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                'Expect that the model is a subclass from padertorch.Module.\n'
                f'Got: type: {type(model)}\n{model}'
            )
        self.model = model

        if isinstance(optimizer, dict):
            # Special case see Janek's example
            # TODO: Hint to example
            model_keys = set(deflatten(model.state_dict(), maxdepth=1).keys())
            assert model_keys == set(optimizer.keys()), (model_keys, optimizer)
            optimizer = optimizer.copy()
            for key, opti in list(optimizer.items()):
                if opti is None:
                    del optimizer[key]
                else:
                    assert isinstance(opti, Optimizer), opti
                    m = getattr(model, key)
                    opti.set_parameters(m.parameters())
        else:
            assert isinstance(optimizer, Optimizer), optimizer
            optimizer.set_parameters(model.parameters())

        self.optimizer = optimizer

        self.device = None  # Dummy value, will be set in Trainer.train

        self.storage_dir = Path(storage_dir).expanduser().resolve()
        self.timer = ContextTimerDict()
        self.reset_timer()
        self.iteration = None
        self.epoch = None

        self.summary_trigger = summary_trigger
        self.checkpoint_trigger = checkpoint_trigger
        self.keep_all_checkpoints = keep_all_checkpoints
        self.max_trigger = max_trigger

        self.loss_weights = loss_weights
        self.virtual_minibatch_size = virtual_minibatch_size

    def reset_timer(self):
        self.timer.clear()

    def test_run(
            self,
            train_iterator,
            validation_iterator,
            device=0 if torch.cuda.is_available() else 'cpu'
    ):
        """
        Run a test on the trainer instance (i.e. model test).

        Also tests weather validation step is deterministic.
        !!Does not work with layers changing their internal state such as BatchNorm!!

        Tests:
         - forward (train and validate)
         - deterministic output in eval
         - simple review dict test

        """
        test_run(
            self,
            train_iterator,
            validation_iterator,
            device=device
        )

    def train(
            self,
            train_iterator,
            validation_iterator=None,
            *,
            hooks=None,
            metrics={'loss': 'min'},
            n_best_checkpoints=1,
            resume=False,
            device=0 if torch.cuda.is_available() else 'cpu'
    ):
        """
        A simplified training loop::

            for epoch in range(1, ...):
                for example in train_iterator:
                    model_out = self.model(example)
                    review = self.model.review(example, model_out)
                    review = maybe_add_loss_from_losses(review)
                    review.backward()
                    self.optimizer.step()
                    add_review_to_tensorboardX(review)

        The remaining code takes care about calling validation and save the
        result to tensorboard (if the a validation_iterator is given), save
        checkpoints, cleanup checkpoints that are stale (not best according
        to metrics and not last) and display a progessbar.
        The code is designed that many aspects can be customized.
        (e.g. test_runtime_tests.py DictTrainer for multi model trainer)

        Args:
            train_iterator:
                The train_iterator is python iterable (e.g. tuple, list, ...)
                that can consumed multiple times (i.e. not generator).

                Usually it will be paderbox.database.BaseIterator that is
                returned from a database in paderbox.database.

            validation_iterator:
                Optional and same type as train_iterator. This iterator is used
                for validation.
            hooks:
                Add additional hooks to the default hooks
                (`Trainer.get_default_hooks`)
            metrics:
                The metrics that are used for the deciding which checkpoint is
                kept. The key is of each entry must be a key 'loss' or a key in
                review['losses'] or review['scalars']. The value indicate if
                the metric has to be maximised ('max') or minimised ('min').
            n_best_checkpoints:
                The numer of checkpoints to keep for each metric. In the moment
                only one checkpoints is supported.
                Use `keep_all_checkpoints=True` from the `__init__` to keep all
                checkpoints.
            resume:
                Whether to resume a training or start a fresh one.
            device:
                Defines the device which shall be used ('cpu', 0, 1, ...).
                If None, the device of the model will not be changed and the
                example to the model is moved to the cpu.
        """
        if resume:
            assert resume is True, resume
            self.load_checkpoint()
        else:
            assert not self.checkpoint_dir.exists(),\
                f'A checkpoint directory already exists. If you want to ' \
                f'restart the training set resume to True.'
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

        # Change model to train mode (e.g. activate dropout)
        self.model.train()

        self.to(device)

        # Reset all gradients
        self.optimizer_zero_grad()

        hooks = self.get_default_hooks(
            hooks,
            train_iterator=train_iterator,
            validation_iterator=validation_iterator,
            metrics=metrics,
            n_best_checkpoints=n_best_checkpoints,
        )

        if self.iteration is None and self.epoch is None:
            self.iteration = 0
            self.epoch = 0
        else:
            # For training continue set the correct last value (default -1)
            for hook in hooks:
                hook.set_last(self.iteration, self.epoch)

        # ================ MAIN TRAINING LOOP! ===================
        try:
            # Count epochs up to infinity if not any stop condition is met. A
            # typical stop condition is a firing `StopTrainingHook`.
            for self.epoch in itertools.count(start=self.epoch):
                epoch_start = True
                for hook in hooks:
                    hook.pre_step(self)

                for self.iteration, example in self.timer(
                    key='time_per_data_loading',
                    iterable=enumerate(
                        train_iterator,
                        start=self.iteration,
                    )
                ):
                    if epoch_start:
                        epoch_start = False
                    else:
                        for hook in hooks:
                            hook.pre_step(self)
                    with self.timer['time_per_train_step']:
                        model_output, review = self.train_step(
                            example,
                            optimize=(self.iteration+1) % self.virtual_minibatch_size == 0,
                        )

                    for hook in hooks:
                        hook.post_step(self, example, model_output, review)

                if self.iteration == 0 and epoch_start:
                    # epoch_start is used to prevent raising an error,
                    # when train_iterator has only one element
                    raise Exception(
                        f'Got an empty train iterator: {train_iterator}'
                    )

                # Fix for next loop
                self.iteration += 1

        except StopTraining:
            pass
        finally:
            try:
                for hook in hooks:
                    hook.close(self)
            except Exception:
                print('Exception in finally. May hide actual exception!!!\n'
                      'You may comment this finally block for debugging.')
                raise

    _start_non_validation_time = None

    def validate(self, validation_iterator):
        """
        used by ValidationHook

        :param validation_iterator:
        :return:
        """
        train_end_time = self.timer.timestamp()

        if self._start_non_validation_time is not None:
            self.timer.timings['non_validation_time'].append(
                train_end_time - self._start_non_validation_time
            )

        # Disable backward mode with `no_grad()`.
        with self.timer['validation_time'], torch.no_grad():
            # Change model to eval mode (e.g. deactivate dropout).
            self.model.eval()
            try:
                for i, example in enumerate(validation_iterator):
                    yield self.validation_step(example)
            finally:
                self.model.train()
                self._start_non_validation_time = self.timer.timestamp()

    def optimizer_zero_grad(self):
        if isinstance(self.optimizer, dict):
            for opti in self.optimizer.values():
                opti.zero_grad()
        else:
            self.optimizer.zero_grad()

    def optimizer_step(self):
        if isinstance(self.optimizer, dict):
            for opti in self.optimizer.values():
                opti.step()
        else:
            self.optimizer.step()

    def train_step(self, example, optimize=True):

        model_out, review = self.step(example)

        with self.timer['time_per_backward']:
            self.backward(review)
            if optimize:
                review = self.clip_grad(review)
                self.optimizer_step()
                self.optimizer_zero_grad()

        return model_out, review

    def validation_step(self, example):
        return self.step(example)

    def step(self, example):
        # TODO: backup OutOfMemory
        with self.timer['time_per_train_step_to_device']:
            example = pt.data.example_to_device(
                example, self.device
            )
        with self.timer['time_per_train_step_forward']:
            model_out = self.model(example)
        with self.timer['time_per_train_step_review']:
            review = self.model.review(example, model_out)
        return model_out, self._maybe_add_loss_to_review(review)

    def _maybe_add_loss_to_review(self, review):
        if 'losses' in review:
            assert 'loss' not in review, review
            losses = review['losses']

            loss = 0.
            loss_weights = self.loss_weights
            if len(losses) != 1:
                if loss_weights is None:
                    raise Exception(
                        'You can not have multiple losses without specifying '
                        f'loss_weights. losses: {losses}'
                    )
                elif set(loss_weights.keys()) != set(losses.keys()):
                    import textwrap
                    from IPython.lib.pretty import pretty
                    raise Exception(
                        'You can not have multiple losses without specifying '
                        'a loss_weight for each loss.'
                        f'\nlosses:'
                        f'\n{textwrap.indent(pretty(losses), " "*4)}'
                        f'\nloss_weights:\n'
                        f'{textwrap.indent(pretty(loss_weights), " "*4)}'
                    )

            for key, value in losses.items():
                weight = loss_weights[key] if loss_weights is not None else 1.
                loss = loss + (weight * value)
                if 'scalars' not in review:
                    review['scalars'] = {}
                review['scalars'][f'{key}_loss_weight'] = weight
            review['loss'] = loss
        else:
            assert 'loss' in review, review

        assert review['loss'].dim() == 0, review['loss']
        assert torch.isfinite(review['loss']), review
        return review

    def backward(self, review, retain_graph=False):
        review['loss'].backward(retain_graph=retain_graph)

    def get_default_hooks(
            self,
            hooks,
            *,
            train_iterator,
            validation_iterator,
            metrics,
            n_best_checkpoints,
    ):
        if n_best_checkpoints != 1:
            raise NotImplementedError(
                f'The implementation for more than one checkpoint is not'
                f'finished.\n'
                f'Requested number of checkponts: {n_best_checkpoints}'
            )

        if hooks is None:
            hooks = []
        try:
            max_it_len = len(train_iterator)
        except TypeError:
            # TypeError: object of type '...' has no len()
            max_it_len = None
        hooks = pt.utils.to_list(hooks)

        writer = tensorboardX.SummaryWriter(str(self.storage_dir))

        if validation_iterator is None:
            print(
                'Since no validation_iterator is provided to `Trainer.train`, '
                'disable validation.'
            )
            raise NotImplementedError(
                'TODO: Check SimpleCheckpointHook for errors'
            )
            hooks.append(SimpleCheckpointHook(
                self.checkpoint_trigger,
                keep_all=self.keep_all_checkpoints,
            ))

            summary_trigger = self.summary_trigger
        else:
            hooks.append(CheckpointedValidationHook(
                trigger=self.checkpoint_trigger,
                iterator=validation_iterator,
                checkpoint_dir=self.checkpoint_dir,
                metrics=metrics,
                keep_all=self.keep_all_checkpoints,
                init_from_json=self.checkpoint_dir.exists(),
                writer=writer,
            ))

            summary_trigger = AnyTrigger(
                self.summary_trigger,
                self.checkpoint_trigger,
            )

        hooks.append(SummaryHook(summary_trigger, writer=writer))
        hooks.append(ProgressBarHook(self.max_trigger, max_it_len))
        hooks.append(StopTrainingHook(self.max_trigger))
        hooks = sorted(hooks, key=lambda h: h.priority, reverse=True)
        return hooks

    def clip_grad(self, summary: dict):
        # TODO: report clipped and unclipped
        # TODO: allow clip=None but still report grad_norm

        summary.setdefault('scalars', {})
        summary.setdefault('histograms', {})

        if isinstance(self.optimizer, dict):
            for key, opti in self.optimizer.items():
                grad_norm = opti.clip_grad()

                summary['scalars'][f'{key}_grad_norm'] = grad_norm
                # underscore was necessary to obtain unique keys to prevent
                # tensorboard error
                summary['histograms'][
                    f'{key}_grad_norm_'] = torch.Tensor([grad_norm])
        else:
            grad_norm = self.optimizer.clip_grad()
            summary['scalars'][f'grad_norm'] = grad_norm
            summary['histograms'][f'grad_norm_'] = \
                torch.Tensor([grad_norm])

        return summary

    @property
    def checkpoint_dir(self):
        return self.storage_dir / 'checkpoints'

    def default_checkpoint_path(self):
        return self.checkpoint_dir / f'ckpt_{self.iteration}.{CKPT_EXT}'

    def state_dict(self):
        if isinstance(self.optimizer, dict):
            optimizer_state_dict = {
                k: opti.state_dict()
                for k, opti in self.optimizer.items()
            }
        else:
            optimizer_state_dict = self.optimizer.state_dict()
            
        return dict(
                model=self.model.state_dict(),
                iteration=self.iteration,
                epoch=self.epoch,
                optimizer=optimizer_state_dict,
        )

    def save_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.default_checkpoint_path()

        torch.save(
            self.state_dict(),
            str(checkpoint_path)
        )

        print(f"{datetime.now()}: Saved model and optimizer state "
              f"at iteration {self.iteration} to {checkpoint_path}")

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        if isinstance(self.optimizer, dict):
            assert set(self.optimizer.keys()) == set(state_dict['optimizer'].keys()), \
                (self.optimizer, state_dict['model'])
            for key, otim in self.optimizer.items():
                otim.load_state_dict(
                    state_dict['optimizer'][key]
                )
        else:
            self.optimizer.load_state_dict(state_dict['optimizer'])

        self.iteration = state_dict['iteration']
        self.epoch = state_dict['epoch']

    def load_checkpoint(self, map_location='cpu'):
        checkpoint_path = self.checkpoint_dir / 'ckpt_latest.pth'
        assert checkpoint_path.is_file(), checkpoint_path

        checkpoint_dict = torch.load(str(checkpoint_path),
                                     map_location=map_location)

        self.load_state_dict(checkpoint_dict)

        print(f"Loaded checkpoint '{checkpoint_path}' (iteration {self.iteration})")

    def to(self, device):
        if device is None:
            # Do nothing
            return

        assert device == 'cpu' or isinstance(device, int), device
        self.model.to(device)
        if isinstance(self.optimizer, dict):
            for key in self.optimizer.keys():
                self.optimizer[key].to(device)
        else:
            self.optimizer.to(device)
        self.device = device

    def cpu(self):
        return self.to('cpu')

    def cuda(self, device=None):
        assert device is None or isinstance(device, int), device
        if device is None:
            device = torch.device('cuda')
        return self.to(device)


class MultiDeviceTrainer(Trainer):
    """

    A Trainer that does not change the model device.
    The losses may be located on different devices, so this trainer moves all
    losses to the cpu.

    Note: The device argument of the Trainer.train is used to move the example
          to the device.
    """

    def _maybe_add_loss_to_review(self, review):
        if 'losses' in review:
            review['losses'] = {
                k: v.cpu()
                for k, v in review['losses'].items()
            }
        return super()._maybe_add_loss_to_review(review)

    def to(self, device):
        pass


class ContextTimerDict:
    """
    To be able to keep the measurements, we need to create the object before.
    Then each measurement can be started with a context manager.

    >>> np.set_printoptions(precision=2)
    >>> timer = ContextTimerDict()
    >>> with timer['test']:
    ...     time.sleep(0.1)
    >>> with timer['test']:
    ...     time.sleep(0.1)
    >>> with timer['test_2']:
    ...     time.sleep(0.1)
    >>> for _ in timer('test_3', range(3)):
    ...     time.sleep(0.1)

    Ignore timing when an exception is raised
    >>> with contextlib.suppress(Exception), timer['test_2']:
    ...     raise Exception
    >>> timer
    ContextTimerDict: {'test': array([0.1, 0.1]), 'test_2': array([0.1]), 'test_3': array([1.96e-06, 4.80e-06, 3.87e-06])}
    >>> timer.as_dict
    {'test': array([0.1, 0.1]), 'test_2': array([0.1]), 'test_3': array([1.96e-06, 4.80e-06, 3.87e-06])}
"""
    def __init__(self):
        self.timestamp = time.perf_counter  # time.process_time
        self.timings = defaultdict(list)
        self.clear()

    def clear(self):
        self.timings.clear()

    @contextlib.contextmanager
    def __getitem__(self, item):
        assert isinstance(item, str), item
        start = self.timestamp()
        yield
        end = self.timestamp()
        self.timings[item].append(end - start)

    @property
    def as_dict(self):
        return {k: np.array(time) for k, time in self.timings.items()}

    def __repr__(self):
        return f'{self.__class__.__name__}: ' + repr(self.as_dict)

    def __str__(self):
        return str(self.as_dict)

    def __call__(self, key, iterable):
        iterator = iter(iterable)

        class StopIterationIgnoredByContextlib(Exception):
            pass
            # contextlib.contextmanager tries to inform the user with a
            # DeprecationWarning because of PEP 479.
            # The cas here is still conform with PEP 479 (i.e. use __future__
            # import).
            # To suppress the warning, convert StopIteration to this Exception
            # and catch it.

        try:
            while True:
                with self[key]:
                    try:
                        example = next(iterator)
                    except StopIteration:
                        raise StopIterationIgnoredByContextlib
                yield example
        except StopIterationIgnoredByContextlib:
            pass


class InteractiveTrainer(Trainer):
    def __init__(
            self,
            model,
            optimizer,
            loss_weights=None,
            max_trigger=(200, 'epoch'),
            summary_trigger=(50, 'epoch'),
            validation_trigger=None,
    ):
        super().__init__(
            model=model,
            storage_dir='this/is/no/path',
            optimizer=optimizer,
            loss_weights=loss_weights,
            summary_trigger=summary_trigger,
            checkpoint_trigger=None,
            keep_all_checkpoints=False,
            max_trigger=max_trigger,
        )
        # Trainer uses checkpoint_trigger as validation_trigger
        if validation_trigger is None:
            self.validation_trigger = summary_trigger
        else:
            self.validation_trigger = validation_trigger

    def get_default_hooks(
            self,
            hooks,
            *,
            train_iterator,
            validation_iterator=None,
            metrics,
            n_best_checkpoints,
    ):
        if n_best_checkpoints != 1:
            raise NotImplementedError(
                f'The implementation for more than one checkpoint is not'
                f'finished.\n'
                f'Requested number of checkponts: {n_best_checkpoints}'
            )

        if hooks is None:
            hooks = []
        else:
            hooks = pt.utils.to_list(hooks)

        self.writer = InteractiveWriter()

        if validation_iterator is not None:

            hooks.append(ValidationHook(
                trigger=self.validation_trigger,
                iterator=validation_iterator,
                # checkpoint_dir=self.checkpoint_dir,
                # metrics=metrics,
                # keep_all=self.keep_all_checkpoints,
                # init_from_json=self.checkpoint_dir.exists(),
                writer=self.writer,
            ))

            summary_trigger = AnyTrigger(
                self.summary_trigger,
                self.validation_trigger,
            )
        else:
            summary_trigger = self.summary_trigger

        try:
            max_it_len = len(train_iterator)
        except TypeError:
            # TypeError: object of type '...' has no len()
            max_it_len = None

        hooks.append(SummaryHook(summary_trigger, writer=self.writer))
        hooks.append(ProgressBarHook(self.max_trigger, max_it_len))
        hooks.append(StopTrainingHook(self.max_trigger))
        hooks = sorted(hooks, key=lambda h: h.priority, reverse=True)
        return hooks

    @functools.wraps(Trainer.train)
    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        return self.writer


class InteractiveWriter:
    def __init__(self):
        self.scalars = collections.defaultdict(list)

    def add_scalar(self, tag, scalar_value, global_step, walltime=None):
        if tag.split('/')[0] in ['training_timings', 'validation_timings']:
            return
        print(f'{global_step}, {tag}: {scalar_value}')

        walltime = time.time() if walltime is None else walltime
        self.scalars[tag].append({
            'value': scalar_value,
            'global_step': global_step,
            'walltime': walltime,
        })

    def add_audio(self, tag, snd_tensor, global_step,
                  sample_rate=44100, walltime=None):
        pass

    def add_image(self, tag, img_tensor, global_step, walltime=None):
        pass

    def add_histogram(self, tag, values, global_step,
                      bins='tensorflow', walltime=None):
        pass

    def close(self):
        pass

# TODO: write function for those to functions outside of trainer
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
