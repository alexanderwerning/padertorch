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

storage_dir = str(
    Path(os.environ['STORAGE_ROOT']) / 'voice_activity' / timeStamped('')[1:]
)
os.makedirs(storage_dir, exist_ok=True)

DEBUG = False
DATA_TEST = False


def get_datasets():
    db = Fearless()
    train = db.get_dataset_train(subset='stream')
    validate = db.get_dataset_validation(subset='stream')

    def prepare_example(example):
        example['audio_path'] = example['audio_path']['observation']
        example['activity'] = db.get_activity(example)
        return example

    train = train.map(prepare_example)
    validate = validate.map(prepare_example)

    training_data = prepare_dataset(train, training=True)
    validation_data = prepare_dataset(validate, training=False)
    return training_data, validation_data


def prepare_dataset(dataset, training=False):
    batch_size = 8#24

    def chunker(example):
        """Split stream into 4s segments"""
        # 4s at 8kHz -> 32k samples
        chunk_length = 32000
        # chunk_count = int(example['num_samples']/chunk_length)
        # if DEBUG:
        #     chunk_count = min(chunk_count, 20)
        # chunks = []
        # for chunk_id in range(chunk_count):
        #     chunk = {'example_id': example['example_id']+'_'+str(chunk_id)}
        #     num_samples = chunk_length

        #     if chunk_id == chunk_count - 1:
        #         num_samples = example['num_samples'] - (chunk_count-1) * chunk_length

        #     start = chunk_id*chunk_length
        #     end = start + num_samples
        #     chunk.update(num_samples=num_samples)
        #     chunk.update(audio_start_samples=start)
        #     chunk.update(audio_stop_samples=end)
        #     chunk.update(audio_path=example['audio_path'])
        #     chunk.update(activity=example['activity'][start:end])
        #     chunks.append(chunk)
        #     np.random.shuffle(chunks)
        # return chunks
        start = max(0, np.random.randint(example['num_samples'])-chunk_length)
        stop = start + chunk_length
        example.update(num_samples=chunk_length)
        example.update(audio_start_samples=start)
        example.update(audio_stop_samples=stop)
        example.update(audio_path=example['audio_path'])
        example.update(activity=example['activity'][start:stop])
        return example

    def select_speech(example):
        """Cut out a section with speech for evaluation.

        We evaluate the model on 30s audio segments which contain speech."""
        first_speech = example['activity'].intervals[0][0]
        max_time_buffer = 8000 * 15 # 15s
        time_buffer = np.random.randint(max_time_buffer)
        length = 8000 * 30 # 30s
        start = max(0, first_speech-time_buffer)
        stop = start + length
        example['audio_start_samples'] = start
        example['audio_stop_samples'] = stop
        example['activity'] = example['activity'][start:stop]
        return example

    if training:
        dataset = dataset.shuffle(reshuffle=True)

    dataset = dataset.prefetch(
        num_workers=8, buffer_size=10*batch_size
    )

    if training:
        dataset = dataset.map(chunker)
    else:
        dataset = dataset.map(select_speech)

    audio_reader = AudioReader(
        source_sample_rate=8000, target_sample_rate=8000
    )
    dataset = dataset.map(audio_reader)

    STFT_SHIFT = 80
    STFT_WINDOW_LENGTH = 400
    STFT_SIZE = 512
    STFT_PAD = True

    stft = STFT(
        shift=STFT_SHIFT,
        size=STFT_SIZE,
        window_length=STFT_WINDOW_LENGTH,
        pad=STFT_PAD,
        fading='half'# was None
    )

    def segment(array):
        frames = int(array.shape[0]/STFT_SHIFT)
        output = np.zeros(frames)
        for i in range(frames):
            middle = i*STFT_SHIFT
            start = max(0, middle-STFT_WINDOW_LENGTH)
            stop = min(middle+STFT_WINDOW_LENGTH, array.shape[0]-1)
            output[i] = array[start: stop].any()
        return output

    def calculate_stft(example):
        complex_spectrum = stft(example['audio_data'].flatten())
        spectrum_magnitude = np.abs(complex_spectrum)**2
        real_magnitude = spectrum_magnitude.astype(np.float32)
        real_magnitude = real_magnitude[None, None, ...]
        example['features'] = rearrange(real_magnitude,
                                        'b c f t -> b c t f', b=1, c=1)[:, :, :-1, :]
        example['activity'] = segment(example['activity'])
        # example['activity'] = segment_axis(example['activity'],
        #                                    length=STFT_WINDOW_LENGTH,
        #                                    shift=STFT_SHIFT,
        #                                    end='pad' if STFT_PAD else 'cut'
        #                                    ).any(axis=-1)
        return example

    dataset = dataset.map(calculate_stft)

    def finalize(example):
        return {
            'example_id': example['example_id'],
            'features': Variable(torch.from_numpy(example['features'])),
            'seq_len': example['features'].shape[-1],
            'activity': example['activity'][:].astype(np.float32)
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
                  pool_size=[1, (4, 1)] + 2*[1, (8, 1)])
    temporal_layer = CNN1d(in_channels=64,
                           out_channels=[128, 10],
                           kernel_size=3,
                           pool_size=1)
    # we cannot pool across the channels using CNN1d
    pooling = MaxPool2d(kernel_size=(10, 1))
    sigmoid = torch.nn.Sigmoid()
    model = SAD_Classifier(cnn2d, temporal_layer, pooling, sigmoid)
    return model


def train(model):
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
    if DATA_TEST:
        train_set, validate_set = get_datasets()
        element = train_set.__iter__().__next__()
        #element['activity'] = element['activity']
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
        train(model)


if __name__ == '__main__':
    main()
