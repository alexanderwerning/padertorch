import dataclasses
import torch
from padertorch.contrib.je.data.transforms import MultiHotAlignmentEncoder
from pb_sed.data_preparation.transform import Transform
from paderbox.transform import resample_sox
import numpy as np
import samplerate
from torch import autocast
from pb_sed.data_preparation.provider import DataProvider
from nt_paths import json_root

from padertorch import Configurable, Trainer

class MobileNetWrapper(torch.nn.Module):
    def __init__(self, model, feature_extractor=None):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        # some models return a tuple (output, hidden_state)

    def forward(self, batch):
        if self.feature_extractor is not None:
            float_input = batch['audio_data'].squeeze(1).float()
            mel_spec = self.feature_extractor(float_input)[:, None]
        else:
            mel_spec = batch['audio_data'].squeeze(1).float()
        
        with autocast(mel_spec.device.type):  # does this break on cpu?
            x, features = self.model(mel_spec)
        return x, features

@dataclasses.dataclass
class RawWaveformTransform(Transform):
    label_encoder: MultiHotAlignmentEncoder
    sample_rate: int = None
    in_rate: int = 16000

    def __call__(self, example):
        example['audio_data'] = np.squeeze(example['audio_data'])
        if example['audio_data'].ndim == 1:
            example['audio_data'] = example['audio_data'][None]
        # example['audio_data'] = resample_sox(example['audio_data'], in_rate=16_000, out_rate=self.sample_rate)
        # faster:
        if self.sample_rate is not None and self.sample_rate != self.in_rate:
            example['audio_data'] = samplerate.resample(example['audio_data'].T, self.sample_rate / self.in_rate, 'sinc_best').T
        
        weak_labels = [
            (0, 1, self.label_encoder.encode(event_label))
            for event_label in example[self.label_encoder.label_key]
        ]
        weak_targets = self.label_encoder.encode_alignment(
            weak_labels, seq_len=1)[0]
        
        if 'label_weights' in example:
            for label, weight in zip(weak_labels, example['label_weights']):
                weak_targets[label[-1]] *= weight

        example_ = {
            'dataset': example['dataset'],
            'example_id': example['example_id'],
            'audio_data': example['audio_data'],
            'seq_len': 1,
            'weak_targets': weak_targets,
        }
        return example_

@dataclasses.dataclass
class ESC50Provider(DataProvider):

    def __post_init__(self):
        super().__post_init__()
    
    def set_validation_fold(self, validation_fold):
        assert validation_fold in range(1,6), validation_fold
        self.validate_set = f'fold0{validation_fold}'
        self.train_set = {f"fold0{i}":1 for i in range(1,6) if i != validation_fold}
    
    def get_train_set(self, validation_fold=1, filter_example_ids=None):
        self.set_validation_fold(validation_fold)
        return super().get_train_set(filter_example_ids=filter_example_ids)

    def get_validate_set(self, validation_fold=1, filter_example_ids=None):
        self.set_validation_fold(validation_fold)
        return super().get_validate_set(filter_example_ids=filter_example_ids)

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['json_path'] = json_root+"/esc50.json"
        super().finalize_dogmatic_config(config)
        num_events = 50
        config['train_fetcher']['min_label_diversity_in_batch'] = 0
        # min(
        #     num_events, config['train_fetcher']['batch_size']
        # )
        config['mix_interval'] = 1.5
        config['train_set'] = ['fold02', 'fold03', 'fold04', 'fold05']
        config['validate_set'] = 'fold01'
        config['cached_datasets'] = ['fold01', 'fold02', 'fold03', 'fold04', 'fold05']


class MixUp:
    def __init__(self, beta=2, label_key="weak_targets"):
        self.beta = beta
        self.label_key = label_key

    def __call__(self, components):
        assert len(components) == 2
        weight = np.random.beta(self.beta, self.beta)
        audio_data = [comp['audio_data'] for comp in components]
        max_len = max([s.shape[-1] for s in audio_data])
        audio_data = [np.pad(s, ((0, 0), (0, max_len - s.shape[-1]))) for s in audio_data]
        mixed_audio = weight * audio_data[0] + (1 - weight) * audio_data[1]
        mix = {
            'example_id': '+'.join([comp['example_id'] for comp in components]),
            'dataset': '+'.join(sorted(set([comp['dataset'] for comp in components]))),
            'audio_data': mixed_audio,
            'seq_len': mixed_audio.shape[1],
            'weak_targets': (weight * components[0][self.label_key]
            + (1 - weight) * components[1][self.label_key]),  # add mix weights!
            # 'label_weights': np.array([weight, 1 - weight]),
        }
        return mix


def prepare(data_provider, trainer, validation_fold=1):
    data_provider = Configurable.from_config(data_provider)

    data_provider.train_transform.label_encoder.initialize_labels(
        labels=['brushing_teeth', 'clapping', 'clock_tick', 'cat', 'church_bells', 'crow', 'washing_machine', 'dog', 'snoring', 'coughing', 'mouse_click', 'crackling_fire', 'laughing', 'clock_alarm', 'can_opening', 'fireworks', 'frog', 'helicopter', 'glass_breaking', 'thunderstorm', 'pig', 'rain', 'siren', 'chirping_birds', 'engine', 'water_drops', 'chainsaw', 'door_wood_creaks', 'sea_waves', 'pouring_water', 'train', 'hand_saw', 'vacuum_cleaner', 'breathing', 'insects', 'wind', 'sneezing', 'rooster', 'crickets', 'crying_baby', 'keyboard_typing', 'footsteps', 'airplane', 'toilet_flush', 'car_horn', 'sheep', 'door_wood_knock', 'cow', 'hen', 'drinking_sipping']
    )
    data_provider.test_transform.label_encoder.initialize_labels()

    trainer = Trainer.from_config(trainer)
    trainer.model.label_mapping = []
    for idx, label in sorted(data_provider.train_transform.label_encoder.inverse_label_mapping.items()):
        assert idx == len(trainer.model.label_mapping), (idx,
                                                         label, len(trainer.model.label_mapping))
        trainer.model.label_mapping.append(label.replace(', ', '__').replace(
            ' ', '').replace('(', '_').replace(')', '_').replace("'", ''))

    train_set = data_provider.get_train_set(validation_fold=validation_fold)
    validate_set = data_provider.get_validate_set(validation_fold=validation_fold)

    return data_provider, trainer, train_set, validate_set


def lr_sched(warm_up_len, ramp_down_start, ramp_down_len, last_lr_value):
    def wrapper(epoch):
        if epoch < warm_up_len:
            return epoch / warm_up_len
        elif epoch < ramp_down_start:
            return 1
        elif epoch < ramp_down_start + ramp_down_len:
            return last_lr_value + (1 - last_lr_value) * (ramp_down_len - (epoch - ramp_down_start)) / ramp_down_len
        else:
            return last_lr_value
    return wrapper


from pb_sed.database.audioset.provider import AudioSetProvider
def get_audioset_loader(bs, dset='balanced', workers=1):
    label_encoder = {
        'factory': MultiHotAlignmentEncoder,
        'label_key': 'events',
        'storage_dir': './audioset_events'
    }
    data_provider = {
        'factory': AudioSetProvider,
        'train_set': {
            'balanced_train': 1 if dset == 'balanced' or dset == 'all' else 0,
            'unbalanced_train': 1 if dset == 'unbalanced' or dset == 'all' else 0,
        },
        'audio_reader': {
            'target_sample_rate': 32000,
        },
        'train_transform': {
            'factory': RawWaveformTransform,
            'label_encoder': label_encoder,
        },
        'test_transform': {
            'factory': RawWaveformTransform,
            'label_encoder': label_encoder,
        },
        'train_fetcher': {
            'batch_size': bs,
            'prefetch_workers': workers,
            'drop_incomplete': False,
            'max_padding_rate': .5,
        },
        'min_class_examples_per_epoch': 0.0,
        'storage_dir': './audioset_events',
        'scale_sampling_fn': None,
        'mix_interval': None,
    }
    AudioSetProvider.get_config(data_provider)
    data_provider =  AudioSetProvider.from_config(data_provider)
    data_provider.train_transform.label_encoder.initialize_labels(
        dataset=data_provider.db.get_dataset(list(filter(
            lambda key: data_provider.train_set[key] > 0,
            data_provider.train_set.keys()
        ))),
        verbose=True
    )
    data_provider.test_transform.label_encoder.initialize_labels()
    return data_provider

from padertorch.contrib.je.data.transforms import MultiHotAlignmentEncoder
from padertorch import Configurable
from pathlib import Path

def get_esc50_loader(bs):
    label_encoder = {
                'factory': MultiHotAlignmentEncoder,
                'label_key': 'events',
                'storage_dir': './esc50_events'
                }

    loader_conf = {
            'factory': ESC50Provider,
            'json_path': str(Path(json_root)/"esc50.json"),
            'audio_reader': {
                'target_sample_rate': 32000,
            },
            'train_transform': {
                'factory': RawWaveformTransform,
                'label_encoder': label_encoder,
            },
            'test_transform': {
                'factory': RawWaveformTransform,
                'label_encoder': label_encoder,
            },
            'storage_dir': '.',
            'train_fetcher': {
                'batch_size': bs,
                'prefetch_workers': 1,
            },
            'test_fetcher': {
                'batch_size': bs,
                'prefetch_workers': 1,
            },
        }
    Configurable.get_config(loader_conf)
    loader = Configurable.from_config(loader_conf)
    loader.train_transform.label_encoder.initialize_labels(
            labels=['brushing_teeth', 'clapping', 'clock_tick', 'cat', 'church_bells', 'crow', 'washing_machine', 'dog', 'snoring', 'coughing', 'mouse_click', 'crackling_fire', 'laughing', 'clock_alarm', 'can_opening', 'fireworks', 'frog', 'helicopter', 'glass_breaking', 'thunderstorm', 'pig', 'rain', 'siren', 'chirping_birds', 'engine', 'water_drops', 'chainsaw', 'door_wood_creaks', 'sea_waves', 'pouring_water', 'train', 'hand_saw', 'vacuum_cleaner', 'breathing', 'insects', 'wind', 'sneezing', 'rooster', 'crickets', 'crying_baby', 'keyboard_typing', 'footsteps', 'airplane', 'toilet_flush', 'car_horn', 'sheep', 'door_wood_knock', 'cow', 'hen', 'drinking_sipping']
        )
    loader.test_transform.label_encoder.initialize_labels()
    return loader