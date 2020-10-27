import math
from pathlib import Path

import dlp_mpi
import numpy as np
import sacred
import torch
import lazy_dataset
from paderbox.array import segment_axis
from padercontrib.database.fearless import Fearless
from padertorch.contrib.examples.voice_activity_detection.train import prepare_dataset
from padertorch.contrib.examples.voice_activity_detection.train import get_model
from padertorch.contrib.jensheit.eval_sad import evaluate_model, smooth_vad

ex = sacred.Experiment('VAD Evaluation')

STFT_SHIFT = 80
STFT_WINDOW_LENGTH = 200
STFT_SIZE = 256
SAMPLE_RATE = 8000
SEGMENT_LENGTH = SAMPLE_RATE * 60
BUFFER_SIZE = SAMPLE_RATE//2  # buffer around segments to avoid artifacts
TRAINED_MODEL = True


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
    per_sample = True


def partition_audio(ex):
    num_samples = ex['num_samples']
    index = ex['index']
    start = index * SEGMENT_LENGTH
    stop = min(start + SEGMENT_LENGTH, num_samples)

    ex['audio_start_samples'] = start - BUFFER_SIZE
    ex['audio_stop_samples'] = stop + BUFFER_SIZE
    ex['activity'] = ex['activity'][start:stop]
    return ex


def get_data(ex):
    num_samples = ex['num_samples']
    dict_dataset = {}
    for index in range(math.ceil(num_samples / SEGMENT_LENGTH)):
        sub_ex = ex.copy()
        sub_ex['index'] = index
        sub_ex_id = str(index)
        sub_ex['example_id'] = sub_ex_id
        dict_dataset[sub_ex_id] = sub_ex
    return prepare_dataset(lazy_dataset.new(dict_dataset), partition_audio, batch_size=1)


def get_model_output(ex, model, per_sample, db):
    predictions = []
    sequence_lengths = []
    dataset = get_data(ex)
    for batch in dataset:
        model_out_org = model(batch).detach().numpy()

        with_buffer_per_sample = activity_frequency_to_time(
                                                model_out_org,
                                                stft_window_length=STFT_WINDOW_LENGTH,
                                                stft_shift=STFT_SHIFT)
        model_out = with_buffer_per_sample[...,BUFFER_SIZE:example['num_samples']+BUFFER_SIZE]
        predictions.extend(model_out_org)
    return predictions


def get_binary_classification(model_out, threshold):
    vad = list()
    for prediction in model_out:
        smoothed_vad = smooth_vad(prediction, threshold=threshold)
        vad.append(smoothed_vad)
    return np.concatenate(vad, axis=-1).astype(np.bool)


@ex.automain
def main(model_dir, num_ths, buffer_zone, ckpt, out_dir, subset, per_sample):
    model_dir = Path(model_dir).resolve().expanduser()
    assert model_dir.exists(), model_dir

    model = get_model()
    if TRAINED_MODEL:
        state_dict = torch.load(Path(model_dir/'ckpt_latest.pth'))['model']
        model.load_state_dict(state_dict)
    db = Fearless()
    model.eval()

    def get_target_fn(ex, per_sample):
        per_sample_vad = db.get_activity(ex)[:]
        if per_sample:
            return per_sample_vad
        per_frame_vad = segment_axis(per_sample_vad,
                                     length=STFT_WINDOW_LENGTH,
                                     shift=STFT_SHIFT,
                                     end='pad'
                                     ).any(axis=-1)
        return per_frame_vad

    with torch.no_grad():
        tp_fp_tn_fn = evaluate_model(
            db.get_dataset_validation(subset),
            lambda ex: get_model_output(ex, model, per_sample, db),
            lambda out, th, ex: get_binary_classification(out, th),
            lambda ex: get_target_fn(ex, per_sample),
            num_thresholds=num_ths,
            buffer_zone=buffer_zone
        )
    if dlp_mpi.IS_MASTER:
        if out_dir is None:
            out_dir = model_dir
        else:
            out_dir = Path(out_dir).expanduser().resolve()
        if TRAINED_MODEL:
            output_file = out_dir / f'tp_fp_tn_fn_fearless_{buffer_zone}.txt'
        else:
            output_file = out_dir / f'tp_fp_tn_fn_fearless_{buffer_zone}_no_train.txt'
        output_file.write_text(
            '\n'.join([
                ' '.join([str(v) for v in value]) for value in
                tp_fp_tn_fn.tolist()
            ]))
