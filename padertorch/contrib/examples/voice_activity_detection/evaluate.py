import json
import math
from pathlib import Path

import dlp_mpi
import lazy_dataset
import numpy as np
import sacred
import torch
from paderbox.array import segment_axis
from padercontrib.database.fearless import Fearless
from padertorch.configurable import Configurable
from padertorch.contrib.examples.voice_activity_detection.train import (
    get_model_config, prepare_dataset)
from padertorch.contrib.jensheit.eval_sad import evaluate_model, smooth_vad
from padertorch.data import example_to_device
from padertorch.io import load_config

experiment = sacred.Experiment('VAD Evaluation')


# adapted from padercontrib.database.chime5.database
def activity_frequency_to_time(
        frequency_activity,
        stft_window_length,
        stft_shift,
        time_length=None,
):

    frequency_activity = np.asarray(frequency_activity)

    frequency_activity = np.broadcast_to(
        frequency_activity[..., None], (*frequency_activity.shape, stft_window_length)
    )

    time_activity = np.zeros(
        (*frequency_activity.shape[:-2],
         frequency_activity.shape[-2] * stft_shift + stft_window_length - stft_shift)
    )

    # Get the correct view to time_signal
    time_signal_seg = segment_axis(
        time_activity, stft_window_length, stft_shift, end=None
    )

    time_signal_seg[:] = frequency_activity

    if time_length is not None:
        if time_length == time_activity.shape[-1]:
            pass
        elif time_length < time_activity.shape[-1]:
            delta = time_activity.shape[-1] - time_length
            assert delta < stft_window_length - stft_shift, (delta, stft_window_length, stft_shift)
            time_activity = time_activity[..., :time_length]

        elif time_length > time_activity.shape[-1]:
            delta = time_length - time_activity.shape[-1]
            assert delta < stft_window_length - stft_shift, (delta, stft_window_length, stft_shift)

            time_activity = pad_axis(
                time_activity,
                pad_width=(0, delta),
                axis=-1,
            )
        else:
            raise Exception('Can not happen')
        assert time_length == time_activity.shape[-1], (time_length, time_activity.shape)

    return time_activity


@experiment.config
def config():
    stft_params = {
        'shift': 80,
        'window_length': 200,
        'size': 256,
        'pad': True
    }
    sample_rate = 8000
    segment_length = sample_rate * 60
    buffer_size = sample_rate//2  # buffer around segments to avoid artifacts

    model_dir = '/home/awerning/tmp_storage/voice_activity/2020-09-11-12-28-01'
    out_dir = '/home/awerning/out'
    num_ths = 201
    buffer_zone = 0.5
    ckpt = 'ckpt_latest.pth'
    subset = 'stream'
    ignore_buffer = False


@experiment.capture
def partition_audio(ex, segment_length, buffer_size):
    """Trim a given example to segment assigned to it via its 'index' attribute.
    The segment is extended by a buffer on both sides. Exceeding the example length or its beginning is possible."""
    index = ex['index']
    start = index * segment_length
    stop = start + segment_length

    ex['audio_start_samples'] = start - buffer_size
    ex['audio_stop_samples'] = stop + buffer_size
    return ex


def read_and_pad_audio(ex, audio_reader):
    """Read the audio and fix issues with the padding.
    If a example exceeds the available samples with its buffer, the overlap is padded with zeros to ensure all examples have exactly the same amount of samples.
    """
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


@experiment.capture
def get_data(ex, stft_params, segment_length):
    """Convert a single example with a stream into a new dataset with small segments of the stream to be used for evaluation.
    This function proceeds by duplicating the given example repeatedly, giving each duplicate an incrementing index.
    We then re-use the 'prepare_dataset' function to trim down the duplicated examples to their respective segments via the 'partition_audio' function.
    """
    num_samples = ex['num_samples']
    ex['num_samples']
    dict_dataset = {}
    for index in range(math.ceil(num_samples / segment_length)):
        sub_ex = ex.copy()
        sub_ex['index'] = index
        sub_ex_id = str(index)
        sub_ex['example_id'] = sub_ex_id
        dict_dataset[sub_ex_id] = sub_ex
    return prepare_dataset(lazy_dataset.new(dict_dataset), lambda ex: partition_audio(ex), stft_params, batch_size=1, audio_reader=read_and_pad_audio)


@experiment.capture
def get_model_output(ex, model, db, stft_params, segment_length, buffer_size):
    """Feed each batch from the dataset into the model and convert the results back from frequency to time domain.
    The buffer added when the original stream was partitioned is removed from the model output."""
    predictions = []
    sequence_lengths = []
    dataset = get_data(ex)

    for batch in dataset:
        batch = example_to_device(batch, 'cpu')
        model_out_org = model(batch).detach().numpy()

        with_buffer_per_sample = activity_frequency_to_time(
                                                model_out_org,
                                                stft_window_length=stft_params['window_length'],
                                                stft_shift=stft_params['shift'])
        model_out = with_buffer_per_sample[..., buffer_size:segment_length+buffer_size]
        predictions.extend(model_out)
    return predictions


def get_binary_classification(model_out, threshold):
    """Smoothen the model output to be more forgiving when evaluating it."""
    vad = list()
    for prediction in model_out:
        smoothed_vad = smooth_vad(prediction, threshold=threshold)
        vad.append(smoothed_vad)
    return np.concatenate(vad, axis=-1).astype(np.bool)


@experiment.automain
def main(model_dir, out_dir, num_ths, buffer_zone, ckpt, subset, segment_length):
    # load model config and checkpoint
    model_dir = Path(model_dir).resolve().expanduser()
    assert model_dir.exists(), model_dir
    config_file = (model_dir/"config.json")
    if config_file.exists():
        config = load_config(config_file)
        model_config = config['model']
    else:
        model_config = get_model_config()
    model = Configurable.from_config(model_config)
    state_dict = torch.load(Path(model_dir/'checkpoints'/ckpt), map_location='cpu')['model']
    model.load_state_dict(state_dict)

    model.eval()
    db = Fearless()

    def get_target_fn(ex):
        padded_length = segment_length * (math.ceil(ex['num_samples'] / segment_length))
        per_sample_vad = np.zeros(padded_length)
        per_sample_vad[:ex['num_samples']] = db.get_activity(ex)[:]
        return per_sample_vad

    # evaluation
    with torch.no_grad():
        tp_fp_tn_fn = evaluate_model(
            db.get_dataset_validation(subset),
            lambda ex: get_model_output(ex, model, db),
            lambda out, th, ex: get_binary_classification(out, th),
            lambda ex: get_target_fn(ex),
            num_thresholds=num_ths,
            buffer_zone=buffer_zone
        )

    # write results to file
    if dlp_mpi.IS_MASTER:
        if out_dir is None:
            out_dir = model_dir
        else:
            out_dir = Path(out_dir).expanduser().resolve()
        model_name = Path(model_dir).stem
        output_file = out_dir / f'stats_fearless_{model_name}_{buffer_zone}.txt'
        output_file.write_text(
            '\n'.join([
                ' '.join([str(v) for v in value]) for value in
                tp_fp_tn_fn.tolist()
            ]))
