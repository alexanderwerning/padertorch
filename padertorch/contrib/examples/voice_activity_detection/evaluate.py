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
from padertorch.contrib.jensheit.eval_sad import evaluate_model

ex = sacred.Experiment('VAD Evaluation')

STFT_SHIFT = 80
STFT_LENGTH = 400
SEGMENT_LENGTH = 8000 * 60
TRAINED_MODEL = False



@ex.config
def config():
    model_dir = '/home/awerning/tmp_storage/voice_activity/2020-09-11-12-28-01/checkpoints'
    out_dir = '/home/awerning/out'
    num_ths = 201
    buffer_zone = 0.0
    ckpt = 'ckpt_latest.pth'
    subset = 'stream'
    ignore_buffer = False
    norm = False
    per_sample = True # determines whether evaluation works on samples or frames (determined by stft)


def partition_audio(ex):
    num_samples = ex['num_samples']
    index = ex['index']
    start = max(index * SEGMENT_LENGTH-2*STFT_SHIFT, 0)
    stop = min((index+1) * SEGMENT_LENGTH+2*STFT_SHIFT, num_samples)
    ex['audio_start_samples'] = start
    ex['audio_stop_samples'] = stop
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


def get_model_output(ex, model, per_sample):
    predictions = []
    sequence_lengths = []
    dataset = get_data(ex)
    for batch in dataset:
        model_out_org = model(batch).detach().numpy()
        if per_sample:
            outputs_ex = []
            for model_out_ex in model_out_org:
                # convolve with STFT_LENGTH/STFT_SHIFT box window #intuition: once one overlapping frame is active, a sample should be active
                model_out_conv = np.convolve(model_out_ex, np.ones(int(STFT_LENGTH / STFT_SHIFT)), )
                # 4 = STFT_LENGTH/STFT_SHIFT -1
                model_out_padded = np.zeros(int(model_out_conv.shape+STFT_LENGTH / STFT_SHIFT - 1))
                model_out_padded[2:-2] = model_out_conv
                model_out_padded[:2] = model_out_conv[0]
                model_out_padded[-2:] = model_out_conv[-1]
                # scale up by STFT_SHIFT
                model_out_per_sample_ex = np.repeat(model_out_padded, STFT_SHIFT)
                model_out_ex = model_out_per_sample_ex
                outputs_ex.append(model_out_ex)
            model_out = np.stack(outputs_ex)
        else:
            model_out = model_out_org

        predictions.extend(model_out)
        sequence_lengths.extend(batch['seq_len'])
    return list(zip(predictions, sequence_lengths))


def get_binary_classification(model_out, threshold):
    vad = list()
    for prediction, seq_len in model_out:
        binarized_prediction = prediction > threshold
        vad.append(binarized_prediction)
    return np.concatenate(vad, axis=-1)


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
        else:
            per_frame_vad = segment_axis(per_sample_vad,
                                         length=400,
                                         shift=80,
                                         end='pad'
                                         ).any(axis=-1)
            return per_frame_vad

    with torch.no_grad():
        tp_fp_tn_fn = evaluate_model(
            db.get_dataset_validation(subset),
            lambda ex: get_model_output(ex, model, per_sample),
            lambda out, th: get_binary_classification(out, th),
            lambda ex: get_target_fn(ex, per_sample),
            num_thresholds=num_ths,
            buffer_zone=0
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
