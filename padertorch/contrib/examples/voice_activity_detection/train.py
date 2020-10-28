#dataset iterator
#map pre-batch operations, stft etc
# chunker
#shuffle
# batch(batch_size)
"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.voice_activity_detection.train
"""
import os
import math
from pathlib import Path

import numpy as np
from paderbox.utils.timer import timeStamped
from paderbox.transform.module_stft import STFT
from padercontrib.database.fearless import Fearless
from padertorch import Trainer
from padertorch.contrib.examples.voice_activity_detection.model import SAD_Classifier
from padertorch.contrib.je.data.transforms import AudioReader, Normalizer, Collate
from padertorch.contrib.je.modules.conv import CNN1d, CNN2d
from padertorch.modules.fully_connected import fully_connected_stack
from padertorch.train.optimizer import Adam
import torch
from torch.nn import MaxPool2d
from torch.autograd import Variable
from paderbox.array import segment_axis
from einops import rearrange

DEBUG = False
DATA_TEST = False

STFT_SHIFT = 80
STFT_WINDOW_LENGTH = 200
STFT_SIZE = 256
STFT_PAD = True
SAMPLE_RATE = 8000


def get_datasets():
    db = Fearless()
    train = db.get_dataset_train(subset='stream')
    validate = db.get_dataset_validation(subset='stream')

    training_data = prepare_dataset(train, chunker, shuffle=True)
    validation_data = prepare_dataset(validate, select_speech, batch_size=2, buffer_factor=2, num_workers=2)
    return training_data, validation_data


def chunker(example):
    """Cut out a random 4s segment from the stream for training."""
    # 4s at 8kHz -> 32k samples
    chunk_length = 4 * SAMPLE_RATE
    start = max(0, np.random.randint(example['num_samples'])-chunk_length)
    stop = start + chunk_length
    example.update(audio_start_samples=start)
    example.update(audio_stop_samples=stop)
    example.update(audio_path=example['audio_path'])
    example.update(activity=example['activity'][start:stop])
    return example


def select_speech(example):
    """Cut out a section with speech for evaluation.

    We evaluate the model on 30s audio segments which contain speech.
    """
    first_speech = example['activity'].intervals[0][0]
    max_time_buffer = 8000 * 1 # 1s
    time_buffer = np.random.randint(max_time_buffer)
    length = 8000 * 30  # 30s
    start = max(0, first_speech-time_buffer)
    stop = start + length
    example['audio_start_samples'] = start
    example['audio_stop_samples'] = stop
    example['activity'] = example['activity'][start:stop]
    return example


def prepare_dataset(dataset, audio_segmentation, shuffle=False, batch_size=8, buffer_factor=4, num_workers=8):
    db = Fearless()

    def prepare_example(example):
        example['audio_path'] = example['audio_path']['observation']
        example['activity'] = db.get_activity(example)
        return example

    dataset = dataset.map(prepare_example)

    if shuffle:
        dataset = dataset.shuffle(reshuffle=True)

    buffer_size = int(buffer_factor * batch_size)
    dataset = dataset.prefetch(
        num_workers=min(num_workers, buffer_size), buffer_size=buffer_size
    )

    dataset = dataset.map(audio_segmentation)

    audio_reader = AudioReader(
        source_sample_rate=8000, target_sample_rate=8000
    )

    def read_and_pad_audio(ex):
        ex_num_samples = ex['audio_stop_samples'] - ex['audio_start_samples']
        padding_front = abs(min(0, ex['audio_start_samples']))
        padding_back = max(0, ex['audio_stop_samples']-ex['num_samples'])
        ex['audio_start_samples'] = max(0, ex['audio_start_samples'])
        ex['audio_stop_samples'] = min(ex['audio_stop_samples'], ex['num_samples'])
        ex = audio_reader(ex)
        padded_audio_data = np.zeros((ex_num_samples))
        if padding_back == 0:
            padded_audio_data[padding_front:] = ex['audio_data'].flatten()
        else:
            padded_audio_data[padding_front:-padding_back] = ex['audio_data'].flatten()
        ex['audio_data'] = padded_audio_data

        return ex

    dataset = dataset.map(read_and_pad_audio)

    stft = STFT(
        shift=STFT_SHIFT,
        size=STFT_SIZE,
        window_length=STFT_WINDOW_LENGTH,
        pad=STFT_PAD,
        fading=None
    )

    def calculate_stft(example):
        complex_spectrum = stft(example['audio_data'])
        real_magnitude = (np.abs(complex_spectrum)**2).astype(np.float32)
        features = rearrange(real_magnitude[None, None, ...],
                             'b c f t -> b c t f', c=1, b=1)[:, :, :-1, :]
        example['features'] = features
        example['activity_samples'] = example['activity'][:]
        example['activity'] = segment_axis(example['activity'][:],
                                           length=STFT_WINDOW_LENGTH,
                                           shift=STFT_SHIFT,
                                           end='pad' if STFT_PAD else 'cut'
                                           ).any(axis=-1)
        return example

    dataset = dataset.map(calculate_stft)

    def finalize(example):
        return {
            'example_id': example['example_id'],
            'features': Variable(torch.from_numpy(example['features'])),
            'seq_len': example['features'].shape[-1],
            'activity': example['activity'][:].astype(np.float32),
            'activity_samples': example['activity_samples'][:].astype(np.float32)
        }

    dataset = dataset.map(finalize)

    dataset = dataset.batch(batch_size).map(Collate(to_tensor=True))

    def unpack_tensor(batch):
        batch['features'] = Variable(torch.from_numpy(np.vstack(batch['features'])))
        return batch

    dataset = dataset.map(unpack_tensor)
    return dataset


def get_model():
    cnn2d = CNN2d(in_channels=1,
                  out_channels=2*[16] + 2*[32] + 2*[64],
                  kernel_size=3,
                  norm='batch',
                  output_layer=False,
                  pool_size=[1, (4, 1)] + 2*[1, (8, 1)])
    temporal_layer = CNN1d(in_channels=64,
                           out_channels=[128, 10],
                           kernel_size=3,
                           input_layer=False,
                           norm='batch',
                           output_layer=False,
                           pool_size=1)
    # we cannot pool across the channels using CNN1d
    pooling = MaxPool2d(kernel_size=(10, 1))
    sigmoid = torch.nn.Sigmoid()
    model = SAD_Classifier(cnn2d, temporal_layer, pooling, sigmoid)
    return model


def train(model, storage_dir):
    train_set, validate_set = get_datasets()
    stop_trigger = 50000
    if DEBUG:
        stop_trigger = 5000
    trainer = Trainer(
        model=model,
        optimizer=Adam(lr=1e-3),
        storage_dir=str(storage_dir),
        summary_trigger=(100, 'iteration'),
        checkpoint_trigger=(1000, 'iteration'),
        stop_trigger=(stop_trigger, 'iteration')
    )
    trainer.register_validation_hook(validate_set)
    trainer.test_run(train_set, validate_set)
    trainer.train(train_set)


def main():
    storage_dir = str(
    Path(os.environ['STORAGE_ROOT']) / 'voice_activity' / timeStamped('')[1:]
    )
    os.makedirs(storage_dir, exist_ok=True)
    if DATA_TEST:
        train_set, validate_set = get_datasets()
        element = train_set.__iter__().__next__()
        model = get_model()
        print(element['features'].shape, element['activity'].shape)
        output = model.forward(element)
        print(output.shape)
        print(model.review(element, output))
        element = validate_set.__iter__().__next__()
        output = model.forward(element)
        print(output.shape, element['activity'].shape)
    else:
        model = get_model()
        train(model, storage_dir)


if __name__ == '__main__':
    main()
