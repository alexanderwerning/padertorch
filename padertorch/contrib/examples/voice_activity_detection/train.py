"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.voice_activity_detection.train
"""
import json
import os
import random
from copy import deepcopy
from pathlib import Path

import lazy_dataset
import numpy as np
import soundfile
import torch
from einops import rearrange
from paderbox.array import segment_axis
from paderbox.transform.module_stft import STFT
from paderbox.utils.timer import timeStamped
from padercontrib.database.fearless import Fearless
from padertorch import Trainer
from padertorch.contrib.examples.voice_activity_detection.model import \
    SAD_Classifier
from padertorch.contrib.je.modules.conv import CNN1d, CNN2d
from padertorch.data.segment import get_segment_boundaries
from padertorch.data.utils import collate_fn
from padertorch.io import dump_config, get_new_storage_dir, load_config
from padertorch.train.optimizer import Adam
from sacred import SETTINGS, Experiment
from sacred.observers.file_storage import FileStorageObserver
from torch.nn import MaxPool2d

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

experiment_name = "sad"
experiment = Experiment(experiment_name)


def get_model_config():
    return load_config("default_model_config.json")


@experiment.config
def config():
    trainer_config = {
        "optimizer": {
            "factory": Adam,
            "lr": 1e-3
        },
        "summary_trigger": (100, "iteration"),
        "checkpoint_trigger": (1000, "iteration"),
        "stop_trigger": (50000, "iteration")
    }
    trainer_config["model"] = get_model_config()
    if "storage_dir" not in trainer_config:
        trainer_config["storage_dir"] = get_new_storage_dir(
            experiment_name, id_naming="time")

    stft_params = {
        'shift': 80,
        'window_length': 200,
        'size': 256,
        'pad': True,
        'fading': None
    }
    sample_rate = 8000

    debug = False
    batch_size = 64
    train_chunk_size = 4 * sample_rate
    validate_chunk_size = 30 * sample_rate

    data_subset = "stream"

    load_model_from = None

    trainer_config = Trainer.get_config(trainer_config)


def debug_dataset(dataset):
    """Create a dataset containing only the first element of the given dataset."""
    first_example = dataset[1]  # this example does not immediately start with activity
    dict_dataset = {index: first_example.copy()
                    for index in range(len(dataset))}
    return lazy_dataset.new(dict_dataset)


@experiment.capture
def get_datasets(data_subset, train_chunk_size, validate_chunk_size, stft_params, batch_size, debug):
    """Initialize the dataset objects."""
    db = Fearless()
    train_set = db.get_dataset_train(subset=data_subset)
    validate_set = db.get_dataset_validation(subset=data_subset)

    if debug:
        train_set = debug_dataset(train_set)
        validate_set = debug_dataset(validate_set)

    training_data = prepare_dataset(
        train_set, chunker, stft_params, shuffle=True, batch_size=batch_size, train=True)
    validation_data = prepare_dataset(
        validate_set, select_speech, stft_params, batch_size=8)
    return training_data, validation_data


@experiment.capture
def chunker(example, train_chunk_size, debug):
    """Cut a batch of 4s segments from the stream for training."""
    examples = []

    boundaries = get_segment_boundaries(example['num_samples'], train_chunk_size, train_chunk_size, anchor='random')

    for start, stop in np.nditer(boundaries, op_axes=0):
        example_chunk = example.copy()
        example_chunk.update(audio_start_samples=start)
        example_chunk.update(audio_stop_samples=stop)
        example_chunk.update(activity=example['activity'][start:stop])
        examples.append(example_chunk)

    random.shuffle(examples)

    return examples


@experiment.capture
def select_speech(example, validate_chunk_size, sample_rate, debug, time_buffer_sec=1):
    """Cut out a section with speech for evaluation.

    We evaluate the model on 30s audio segments which contain speech.
    """
    first_speech = example['activity'].intervals[0][0]

    if not debug:
        max_time_buffer = sample_rate * time_buffer_sec  # 1s
        time_buffer = np.random.randint(max_time_buffer)
    else:
        max_time_buffer = sample_rate * time_buffer_sec * 10
        time_buffer = max_time_buffer
    start = max(0, first_speech-time_buffer)
    stop = start + validate_chunk_size
    example['audio_start_samples'] = start
    example['audio_stop_samples'] = stop
    example['activity'] = example['activity'][start:stop]
    return example


def prepare_dataset(dataset, audio_segmentation, stft_params, shuffle=False, batch_size=8, num_workers=6, train=False, audio_reader=None):
    """Apply transformations to a dataset so it can be used for training a model.

        Args:
            dataset:
                The dataset object on which to apply the transformations.
            audio_segmentation:
                A function which reduces the audio stream to a small segment or a batch of segments.
            stft_params:
                The parameters used for the STFT.
            shuffle:
                Whether to shuffle the streams before applying the 'audio_segmentation'. Note that if a stream is segmented into multiple segments, these need to be shuffled by the segmentation function.
            batch_size:
                The size of the resulting batches.
            num_workers:
            train:
                Whether this is a training or a validation dataset. It is assumed that in training 'audio_segmentation' produces batches, whereas in validation it returns single segments per example.
            audio_reader:
                A function to read the audio files. If set to 'None', a default is used, which reads the given 'audio_path' from 'audio_start_samples' to 'audio_stop_samples' into 'audio_data'.
        Returns:
            dataset:
                The dataset with the transformations added.
        """

    def prepare_example(example):
        """Simplify audio_path attribute and load activity data for an example."""
        example['audio_path'] = example['audio_path']['observation']
        example['activity'] = Fearless.get_activity(example)
        return example

    dataset = dataset.map(prepare_example)

    # shuffle the streams
    if shuffle:
        dataset = dataset.shuffle(reshuffle=True)

    # create batches with segments from the streams
    # when evaluating, this does not result in batches, but single examples
    dataset = dataset.map(audio_segmentation)

    # unbatch if we are training (otherwise there are no batches)
    if train:
        dataset = dataset.unbatch()

    # default audio reader
    def read_audio(example):
        """Read the audio from the given path according to the selected samples."""
        audio_path = str(example["audio_path"])
        start_samples = example["audio_start_samples"]
        stop_samples = example["audio_stop_samples"]

        x, sr = soundfile.read(
            audio_path, start=start_samples, stop=stop_samples)
        audio = x.T
        example["audio_data"] = audio
        return example

    # read audio file into examples
    if audio_reader is None:
        dataset = dataset.map(read_audio)
    else:
        dataset = dataset.map(lambda ex: audio_reader(ex, read_audio))

    # apply stft to audio
    stft = STFT(**stft_params)

    def calculate_stft(example):
        """Calculate the STFT spectrum."""
        complex_spectrum = stft(example['audio_data'])
        real_magnitude = (np.abs(complex_spectrum)**2).astype(np.float32)
        features = rearrange(real_magnitude[None, None, ...],
                             'b c t f -> b c f t', c=1, b=1)[:, :, :, :]  # reorder the dimensions to batch | channel | frequency | time
        example['features'] = features
        example['activity_samples'] = example['activity'][:]
        example['activity'] = segment_axis(example['activity'][:],
                                           length=stft_params['window_length'],
                                           shift=stft_params['shift'],
                                           end='pad' if stft_params['pad'] else 'cut'
                                           ).any(axis=-1)
        return example

    dataset = dataset.map(calculate_stft)

    # renaming and typecasting
    def finalize(example):
        return {
            'example_id': example['example_id'],
            'features': example['features'],
            'seq_len': example['features'].shape[-1],
            'activity': example['activity'][:].astype(np.float32),
            'activity_samples': example['activity_samples'][:].astype(np.float32)
        }

    dataset = dataset.map(finalize)

    # create batches and stack vectors
    dataset = dataset.batch(batch_size).map(collate_fn)

    def unpack_tensor(batch):
        batch['features'] = np.vstack(batch['features'])
        batch['activity'] = np.vstack(batch['activity'])
        batch['activity_samples'] = np.vstack(batch['activity_samples'])
        return batch

    dataset = dataset.map(unpack_tensor)

    return dataset


@experiment.capture
def get_trainer(trainer_config, load_model_from):
    """Initialize the PaderTorch Trainer object."""
    trainer = Trainer.from_config(trainer_config)

    checkpoint_path = trainer.checkpoint_dir / 'ckpt_latest.pth'
    if load_model_from is not None and not checkpoint_path.is_file():
        checkpoint = torch.load(load_model_from)
        trainer.model.load_state_dict(checkpoint['model'])

    return trainer


@experiment.capture
def train(train_set, validate_set):
    trainer = get_trainer()
    trainer.register_validation_hook(validate_set)
    trainer.test_run(train_set, validate_set)
    trainer.train(train_set)


@experiment.automain
def main(trainer_config):
    experiment.observers.append(FileStorageObserver(
        Path(trainer_config['storage_dir']) / 'sacred')
    )
    storage_dir = Path(trainer_config['storage_dir'])
    os.makedirs(storage_dir, exist_ok=True)
    train_set, validate_set = get_datasets()
    dump_config(trainer_config, storage_dir/'config.json')
    train(train_set, validate_set)
