import math
from pathlib import Path

import dlp_mpi
import numpy as np
import sacred
import torch
import lazy_dataset
from paderbox.array import segment_axis
from padercontrib.database.fearless import Fearless
from padertorch.configurable import Configurable
from padertorch.contrib.examples.voice_activity_detection.train import prepare_dataset
from padertorch.contrib.examples.voice_activity_detection.train import get_model_config
from padertorch.contrib.jensheit.eval_sad import evaluate_model, smooth_vad


ex = sacred.Experiment('VAD Evaluation')

STFT_SHIFT = 80
STFT_WINDOW_LENGTH = 200
STFT_SIZE = 256
SAMPLE_RATE = 8000
SEGMENT_LENGTH = SAMPLE_RATE * 60
BUFFER_SIZE = SAMPLE_RATE//2  # buffer around segments to avoid artifacts
TRAINED_MODEL = True

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


@ex.config
def config():
    model_dir = '/home/awerning/tmp_storage/voice_activity/2020-09-11-12-28-01/checkpoints'
    out_dir = '/home/awerning/out'
    num_ths = 201
    buffer_zone = 0.5
    ckpt = 'ckpt_latest.pth'
    subset = 'stream'
    ignore_buffer = False
    norm = False


def partition_audio(ex):
    num_samples = ex['num_samples']
    index = ex['index']
    start = index * SEGMENT_LENGTH
    stop = start + SEGMENT_LENGTH

    ex['audio_start_samples'] = start - BUFFER_SIZE
    ex['audio_stop_samples'] = stop + BUFFER_SIZE
    return ex


def get_data(ex):
    num_samples = ex['num_samples']
    ex['num_samples']
    dict_dataset = {}
    for index in range(math.ceil(num_samples / SEGMENT_LENGTH)):
        sub_ex = ex.copy()
        sub_ex['index'] = index
        sub_ex_id = str(index)
        sub_ex['example_id'] = sub_ex_id
        dict_dataset[sub_ex_id] = sub_ex
    return prepare_dataset(lazy_dataset.new(dict_dataset), partition_audio, batch_size=1)


def get_model_output(ex, model, db):
    predictions = []
    sequence_lengths = []
    dataset = get_data(ex)
    for batch in dataset:
        model_out_org = model(batch).detach().numpy()

        with_buffer_per_sample = activity_frequency_to_time(
                                                model_out_org,
                                                stft_window_length=STFT_WINDOW_LENGTH,
                                                stft_shift=STFT_SHIFT)
        model_out = with_buffer_per_sample[..., BUFFER_SIZE:SEGMENT_LENGTH+BUFFER_SIZE]
        predictions.extend(model_out)
    return predictions


def get_binary_classification(model_out, threshold):
    vad = list()
    for prediction in model_out:
        smoothed_vad = smooth_vad(prediction, threshold=threshold)
        vad.append(smoothed_vad)
    return np.concatenate(vad, axis=-1).astype(np.bool)


@ex.automain
def main(model_dir, num_ths, buffer_zone, ckpt, out_dir, subset):
    model_dir = Path(model_dir).resolve().expanduser()
    assert model_dir.exists(), model_dir
    model_file = (model_dir/"model.json")
    if model_file.exists():
        model_config = json.loads(model_file.read_text())
    else:
        model_config = get_model_config()
    model = Configurable.from_config(model_config)
    if TRAINED_MODEL:
        state_dict = torch.load(Path(model_dir/'checkpoints'/'ckpt_latest.pth'))['model']
        model.load_state_dict(state_dict)
    db = Fearless()
    model.eval()

    def get_target_fn(ex):
        padded_length = SEGMENT_LENGTH*(math.ceil(ex['num_samples'] / SEGMENT_LENGTH))
        per_sample_vad = np.zeros(padded_length)
        per_sample_vad[:ex['num_samples']] = db.get_activity(ex)[:]
        return per_sample_vad

    with torch.no_grad():
        tp_fp_tn_fn = evaluate_model(
            db.get_dataset_validation(subset),
            lambda ex: get_model_output(ex, model, db),
            lambda out, th, ex: get_binary_classification(out, th),
            lambda ex: get_target_fn(ex),
            num_thresholds=num_ths,
            buffer_zone=buffer_zone
        )
    if dlp_mpi.IS_MASTER:
        if out_dir is None:
            out_dir = model_dir
        else:
            out_dir = Path(out_dir).expanduser().resolve()
        if TRAINED_MODEL:
            model_name = Path(model_dir).parent.stem
            output_file = out_dir / f'stats_fearless_{model_name}_{buffer_zone}.txt'
        else:
            output_file = out_dir / f'stats_fearless_{buffer_zone}_no_train.txt'
        output_file.write_text(
            '\n'.join([
                ' '.join([str(v) for v in value]) for value in
                tp_fp_tn_fn.tolist()
            ]))
