"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.voice_activity_detection.train
"""
import random
import json
import os
from pathlib import Path

from sacred import Experiment
from sacred.observers.file_storage import FileStorageObserver

import numpy as np
from paderbox.utils.timer import timeStamped
from paderbox.transform.module_stft import STFT
from padercontrib.database.fearless import Fearless
from padertorch import Trainer
from padertorch.io import get_new_storage_dir, dump_config, load_config
from padertorch.contrib.examples.voice_activity_detection.model import SAD_Classifier
from padertorch.contrib.je.modules.conv import CNN1d, CNN2d
from padertorch.train.optimizer import Adam
from padertorch.data.utils import collate_fn
import torch
from torch.nn import MaxPool2d
from paderbox.array import segment_axis
from einops import rearrange

import soundfile

experiment_name = "sad"
experiment = Experiment(experiment_name)


def get_model_config():
    return load_config("default_model_config.json")
    return {
        "factory": SAD_Classifier,
        "conv_layer": {
            "factory": CNN2d,
            "in_channels": 1,
            "out_channels": 2*[16] + 2*[32] + 2*[64],
            "kernel_size": 3,
            "norm": 'batch',
            "output_layer": False,
            "pool_size": [1, (4, 1)] + 2*[1, (8, 1)]
        },
        "temporal_layer": {
            "factory": CNN1d,
            "in_channels": 64,
            "out_channels": [128, 10],
            "kernel_size": [3, 1],
            "input_layer": False,
            "norm": 'batch',
            "output_layer": True,
            "pool_size": 1
        },
        "pooling": {
            "factory": MaxPool2d,
            "kernel_size": (10, 1)
        },
        "activation": {
            "factory": torch.nn.Sigmoid
        }
    }

@experiment.config
def config():
     trainer_config = json_load("default_config.json")
     trainer_config["model"] = get_model_config()
    stft_params = {
        'STFT_SHIFT': 80,
        'STFT_WINDOW_LENGTH': 200,
        'STFT_SIZE': 256,
        'STFT_PAD': True
    }
    sample_rate = 8000

    debug = False
    batch_size = 64
    batches_buffer = 4
    train_chunk_size = 4 * sample_rate
    validate_chunk_size = 30 * sample_rate

    data_subset = "stream"

    load_model_from = None

    trainer_config = {
        "model": get_model_config(),
        "storage_dir": get_new_storage_dir(experiment_name, timeStamped('')[1:]),
        "optimizer": {
            "factory": Adam,
            "lr": 1e-3
        },
        "summary_trigger": (100, "iteration"),
        "checkpoint_trigger": (1000, "iteration"),
        "stop_trigger": (50000, "iteration") if not debug else (5000, "iteration"),
    }


def get_datasets(subset, train_chunk_size, validate_chunk_size, batch_size, batches_buffer, stft_params):
    db = Fearless()
    train_set = db.get_dataset_train(subset=subset)
    validate_set = db.get_dataset_validation(subset=subset)

    training_data = prepare_dataset(train_set, lambda ex: chunker(ex, chunk_size=train_chunk_size), stft_params, shuffle=True, batch_size=batch_size, batches_buffer=batches_buffer, train=True)
    validation_data = prepare_dataset(validate_set, lambda ex: select_speech(ex, chunk_size=validate_chunk_size), stft_params, batch_size=8, batches_buffer=batches_buffer)
    return training_data, validation_data


def chunker(example, chunk_size):
    """Cut out a random 4s segment from the stream for training."""

    start = max(0, np.random.randint(example['num_samples'])-chunk_size)

    examples = []

    for _ in range((example['num_samples']-start)//chunk_size):
        example_chunk = example.copy()
        stop = start + chunk_size
        example_chunk.update(audio_start_samples=start)
        example_chunk.update(audio_stop_samples=stop)
        example_chunk.update(activity=example['activity'][start:stop])
        examples.append(example_chunk)
        start = stop

    random.shuffle(examples)

    return examples


def select_speech(example, chunk_size=30*8000):
    """Cut out a section with speech for evaluation.

    We evaluate the model on 30s audio segments which contain speech.
    """
    first_speech = example['activity'].intervals[0][0]
    max_time_buffer = 8000 * 1 # 1s
    time_buffer = np.random.randint(max_time_buffer)
    start = max(0, first_speech-time_buffer)
    stop = start + chunk_size
    example['audio_start_samples'] = start
    example['audio_stop_samples'] = stop
    example['activity'] = example['activity'][start:stop]
    return example


def prepare_dataset(dataset, audio_segmentation, stft_params, shuffle=False, batch_size=8, batches_buffer=4, num_workers=6, train=False, audio_reader_fn=lambda x: x()):

    def prepare_example(example):
        example['audio_path'] = example['audio_path']['observation']
        example['activity'] = Fearless.get_activity(example)
        return example

    dataset = dataset.map(prepare_example)

    if shuffle:
        dataset = dataset.shuffle(reshuffle=True)

    dataset = dataset.map(audio_segmentation)

    if train:
        dataset = dataset.unbatch()

    buffer_size = int(batches_buffer * batch_size)


    def read_audio(example):
        audio_path = str(example["audio_path"])
        start_samples = example["audio_start_samples"]
        stop_samples = example["audio_stop_samples"]

        x, sr = soundfile.read(
            audio_path, start=start_samples, stop=stop_samples, always_2d=True
        )
        audio = x.T
        example["audio_data"] = audio
        return example

    audio_reader = AudioReader(
        source_sample_rate=8000, target_sample_rate=8000
    )

    dataset = dataset.map(audio_reader_fn(audio_reader))

    stft = STFT(
        shift=stft_params['STFT_SHIFT'],
        size=stft_params['STFT_SIZE'],
        window_length=stft_params['STFT_WINDOW_LENGTH'],
        pad=stft_params['STFT_PAD'],
        fading=None
    )

    def calculate_stft(example):
        complex_spectrum = stft(example['audio_data'])
        real_magnitude = (np.abs(complex_spectrum)**2).astype(np.float32)
        features = rearrange(real_magnitude[None, None, ...],
                             'b c t f -> b c f t', c=1, b=1)[:, :, :, :]
        example['features'] = features
        example['activity_samples'] = example['activity'][:]
        example['activity'] = segment_axis(example['activity'][:],
                                           length=stft_params['STFT_WINDOW_LENGTH'],
                                           shift=stft_params['STFT_SHIFT'],
                                           end='pad' if stft_params['STFT_PAD'] else 'cut'
                                           ).any(axis=-1)
        return example

    dataset = dataset.map(calculate_stft)

    def finalize(example):
        return {
            'example_id': example['example_id'],
            'features': example['features'],
            'seq_len': example['features'].shape[-1],
            'activity': example['activity'][:].astype(np.float32),
            'activity_samples': example['activity_samples'][:].astype(np.float32)
        }

    dataset = dataset.map(finalize)

    dataset = dataset.batch(batch_size).map(collate_fn)

    def unpack_tensor(batch):
        batch['features'] = np.vstack(batch['features'])
        return batch

    dataset = dataset.map(unpack_tensor)

    return dataset


def get_trainer(trainer_config, load_model_from):
    trainer = Trainer.from_config(trainer_config)

    checkpoint_path = trainer.checkpoint_dir / 'ckpt_latest.pth'
    if load_model_from is not None and not checkpoint_path.is_file():
        checkpoint = torch.load(load_model_from)
        trainer.model.load_state_dict(checkpoint['model'])

    return trainer


def train(trainer_config, train_set, validate_set, load_model_from):
    trainer = get_trainer(trainer_config, load_model_from)
    trainer.register_validation_hook(validate_set)
    trainer.test_run(train_set, validate_set)
    trainer.train(train_set)


@experiment.automain
def main(trainer_config, batch_size, train_chunk_size, validate_chunk_size, batches_buffer, data_subset, load_model_from, ):
    Trainer.get_config(trainer_config)

    experiment.observers.append(FileStorageObserver(
        Path(trainer_config['storage_dir']) / 'sacred')
    )
    storage_dir = Path(trainer_config['storage_dir'])
    os.makedirs(storage_dir, exist_ok=True)
    train_set, validate_set = get_datasets(data_subset, train_chunk_size, validate_chunk_size, batch_size, batches_buffer, stft_params)
    dump_config(trainer_config, storage_dir/'config.json')
    train(trainer_config, train_set, validate_set, load_model_from)
