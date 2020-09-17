import math
from pathlib import Path

import numpy as np
import sacred
import torch
import lazy_dataset
from padertorch.contrib.jensheit.batch import Padder 
from padercontrib.database.fearless import Fearless
from padertorch.contrib.examples.voice_activity_detection.train import prepare_dataset
from padertorch.contrib.examples.voice_activity_detection.train import get_model
from padertorch.contrib.jensheit.eval_sad import evaluate_model

ex = sacred.Experiment('VAD Evaluation')


@ex.config
def config():
    model_dir = '/home/awerning/tmp_storage/voice_activity/2020-09-11-12-28-01/checkpoints'
    out_dir = '/home/awerning/out'
    num_ths = 201
    buffer = 0.5
    ckpt = 'ckpt_latest.pth'
    dataset = 'Dev_stream'
    ignore_buffer = False
    norm = False


def partition_audio(ex):
    segment_length = 8000 * 60
    num_samples = ex['num_samples']
    index = ex['index']
    start = max(index * segment_length - 8000, 0)
    stop = min(start + segment_length + 8000, num_samples)
    ex['audio_start_samples'] = start
    ex['audio_stop_samples'] = stop
    ex['activity'] = ex['activity'][start:stop]
    return ex


# divide audio of ex into 61s snippets and use segments long stft # the first 1s is buffer
def get_model_output(ex, model):
    segment_length = 8000 * 60
    num_samples = ex['num_samples']

    predictions = []
    sequence_lengths = []
    dict_dataset = {}
    for index in range(math.ceil(num_samples / segment_length)):
        sub_ex = ex.copy()
        sub_ex['index'] = index
        sub_ex_id = str(index)
        sub_ex['example_id'] = sub_ex_id
        dict_dataset[sub_ex_id] = sub_ex
    dataset = prepare_dataset(lazy_dataset.new(dict_dataset), partition_audio, batch_size=1)
    for batch in dataset:
        model_out_org = model(batch).detach().numpy()
        predictions.extend(model_out_org)
        sequence_lengths.extend(batch['seq_len'])
    return list(zip(predictions, sequence_lengths))


def get_binary_classification(model_out, threshold):
    vad = list()
    start = 0
    for prediction, seq_len in model_out:
        
        binarized_prediction = prediction > threshold
        vad_out = np.repeat(binarized_prediction, 80)
        segment_length = seq_len*80
        # remove buffer
        
        offset = 0 if start == 0 else 8000
        end = start + segment_length
        end = end if end <= num_samples else num_samples
        end_vad = segment_length if end < num_samples else num_samples - start
        vad.append(vad_out[offset:end_vad])
        start += segment_length
    return np.concatenate(vad, axis=-1)


@ex.automain
def main(model_dir, num_ths, buffer, ckpt, out_dir, dataset):
    model_dir = Path(model_dir).resolve().expanduser()
    assert model_dir.exists(), model_dir

    model = get_model()
    state_dict = torch.load(Path(model_dir/'ckpt_latest.pth'))['model']
    model.load_state_dict(state_dict)
    db = Fearless()
    model.eval()

    def get_target_fn(ex):
        return db.get_activity(ex)[:]  # ground truth

    with torch.no_grad():
        tp_fp_tn_fn = evaluate_model(
            db.get_dataset(dataset),
            lambda ex: get_model_output(ex, model),
            lambda out, th: get_binary_classification(out, th),
            get_target_fn, num_thresholds=num_ths
        )

    if out_dir is None:
        out_dir = model_dir
    else:
        out_dir = Path(out_dir).expanduser().resolve()
    (out_dir / f'tp_fp_tn_fn_fearless_{buffer}.txt').write_text(
        '\n'.join([
            ' '.join([str(v) for v in value]) for value in
            tp_fp_tn_fn.tolist()
        ]))
