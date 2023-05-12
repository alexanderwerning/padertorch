import dataclasses
from typing import Optional, Tuple

from einops import rearrange
import nnAudio.Spectrogram
import numpy as np
from paderbox.array import segment_axis
from paderbox.transform.module_fbank import MelTransform
from padertorch.contrib.je.data.transforms import STFT
from paderbox.transform.module_stft import stft_to_spectrogram
import torch
import torch.nn as nn


@dataclasses.dataclass
class RandomTimeCrop:
    crop_size: int = 10

    def __call__(self, data):
        """Crop input randomly.
        Args:
            data: the array to crop (numpy/torch/...)
        """
        # crop to correct length
        # Trim or pad
        # adapted from MSM-MAE implementation
        # deterministic validation steps
        # np.random.seed(hash(str(seed)+("".join(example['example_id'])))%2**32)
        length = data.shape[-1]
        if length > self.crop_size:
            start = np.random.randint(length - self.crop_size)
            data = data[..., start: start + self.crop_size]
        elif length < self.crop_size:
            pad_param = []
            for i in range(len(data.shape)):
                if i == 0:
                    pad_param.insert(0, (0, self.crop_size - length))
                else:
                    pad_param.insert(0, (0, 0))
            data = np.pad(data, pad_param, mode="constant")

        return data


@dataclasses.dataclass
class TimeDomainSegmenter:
    """Segment time-domain audio signal in examples.

    Guarantees a length of segment_length for each segment.

    Args:
        segment_length: segment length in frames
        pad_last: decides if the last, partial segment is padded or thrown out if it exists
    """

    segment_length: int = (
        208 * 160
    )  # int(208 * (160 - 1.5))  # segment length in frames
    # overlapping?
    segment_shift = (
        208 * 160 - 1600
    )  # smallest overlap s.t. 5 segments are created for each 10s ex of audioset
    pad_last: bool = False

    def __call__(self, example):
        audio_data = example["audio_data"]
        if len(audio_data.shape) == 1:
            audio_data = audio_data[None]  # add channel dim if not exists

        segmented = segment_axis(
            audio_data, length=self.segment_length, shift=self.segment_shift
        )

        # move segment dimension to beginning
        rearranged_segments = rearrange(segmented, "c s t -> s c t")

        batch = []
        for i, seg in enumerate(rearranged_segments):
            batch.append(
                {
                    "dataset": example["dataset"],
                    "example_id": f"{example['example_id']}_s{i}",
                    "audio_data": seg,
                    "seq_len": 1,  # dummy
                    "weak_targets": np.zeros(1),  # dummy
                    "strong_targets": np.zeros(1),  # dummy
                    "boundary_targets": np.zeros(1),  # dummy
                }
            )
        return batch


@dataclasses.dataclass
class TimeDomainViTSegmenter:
    """Segment time-domain audio signal in examples for a ViT-like model.

    Ensures that the segment length is a multiple of the patch size.

    Args:
        segment_length: segment length in frames
        pad_last: decides if the last, partial segment is padded or thrown out if it exists

    >>> stft = STFT(size=400, shift=160, fading="half")
    >>> segmenter = TimeDomainViTSegmenter(patch_size=(16,16), allow_shorter_segments=True, stft=stft, max_grid_w=13)
    >>> example = {'audio_data': np.random.randn(1, 10*16000), 'example_id': 'test', 'dataset': 'test'}
    >>> batch = segmenter(example)
    >>> len(batch)
    5
    >>> stft(batch[0]['audio_data']).shape
    (1, 208, 201)
    >>> stft(batch[-1]['audio_data']).shape
    (1, 144, 201)
    """

    pad_last: bool = False
    patch_size: Tuple = (16, 16)
    patch_overlap: Tuple = (0, 0)
    max_grid_w: int = 13
    allow_shorter_segments: bool = False
    stft: Optional[STFT] = None

    def __post_init__(self):
        self.patch_stride = self.patch_size[1] - self.patch_overlap[1]
        num_frames = self.patch_size[1]+(self.max_grid_w-1)*self.patch_stride
        self.segment_length = self.stft.frames_to_samples(num_frames)
        self.segment_shift = self.segment_length
        if self.stft.fading == "half":
            self.segment_shift -= self.stft.shift*(self.stft.window_length//self.stft.shift-1)
        else:
            raise NotImplementedError()

    def compute_segment_end(self, num_samples):
        """Compute the end of the last segment that fits into the audio data.
        Args:
            num_samples: number of samples left in the audio data
        Returns:
            end of the last segment that fits into the audio data
        
        >>> stft = STFT(size=1024, window_length=960, shift=320, fading="half", pad=True)
        >>> segmenter = TimeDomainViTSegmenter(patch_size=(16,16), allow_shorter_segments=True, stft=stft, max_grid_w=13)
        >>> examples = [182020, 158720, 23040, 160000]
        >>> for e in examples:
        ...     s = segmenter.compute_segment_end(158720)
        ...     (stft({"audio_data":torch.zeros(s)})["stft"].shape[0]-segmenter.patch_size[0])%segmenter.patch_stride
        0
        0
        0
        0
        """
        if self.stft.fading == "half":
            padding = self.stft.window_length - self.stft.shift
        elif self.stft.fading == "full":
            padding = 2 * (self.stft.window_length - self.stft.shift)
        else:
            padding = 0
        samples_per_patch = self.patch_size[0]*self.stft.shift
        samples_per_patch_stride = self.patch_stride*self.stft.shift
        remaining_samples = num_samples - samples_per_patch + self.patch_stride
        if remaining_samples <= 0:
            return 0
        num_patches = remaining_samples // samples_per_patch_stride
        return (num_patches - 1) * samples_per_patch_stride + samples_per_patch
    
    def __call__(self, example):
        audio_data = example["audio_data"]
        if len(audio_data.shape) == 1:
            audio_data = audio_data[None]  # add channel dim if not exists

        # number of samples to add to the segment length s.t. a full patch width of new frames is created
        patch_samples = self.patch_size[0]*self.stft.shift
        rearranged_segments = []
        for seg_start in range(0, audio_data.shape[-1], self.segment_shift):
            if seg_start+self.segment_length > audio_data.shape[-1]:
                
                # find segment length such that the number of frames is a multiple of the patch width
                # TODO: is there a better solution? explictly compute the fading frames and avoid looping?
                if self.allow_shorter_segments:
                    seg_len = self.compute_segment_end(audio_data.shape[-1]-seg_start)
                    assert audio_data.shape[-1]-seg_start >= seg_len
                    rearranged_segments.append(
                        audio_data[..., seg_start:seg_start+seg_len])
            else:
                rearranged_segments.append(
                    audio_data[..., seg_start:seg_start+self.segment_length])

        batch = []
        for i, seg in enumerate(rearranged_segments):
            overwrite_keys = "example_id", "audio_data", "seq_len"
            new_example = {k: example[k] for k in example if k not in overwrite_keys}
            new_example["example_id"] = example['example_id']  # f"{example['example_id']}_s{i}" # breaks tuning
            new_example["audio_data"] = seg
            new_example["seq_len"] = seg.shape[-1]
            batch.append(new_example)
        return batch


@dataclasses.dataclass
class MelSpecSegmenter:
    """Segment mel spectrogram in examples.

    Guarantees a length of segment_length for each segment.

    Args:
        segment_length: segment length in frames
        pad_last: decides if the last, partial segment is padded or thrown out if it exists
    """

    segment_length: int = 208  # segment length in frames
    # overlapping?
    pad_last: bool = False

    def __call__(self, example):
        mel_spec = example["mel_spec"]
        num_frames = mel_spec.shape[-1]
        full_segments = num_frames // self.segment_length
        segments = []
        for i in range(full_segments):
            start = i * self.segment_length
            segments.append(mel_spec[..., start: start + self.segment_length])

        last_segment_length = num_frames % self.segment_length
        if self.pad_last and last_segment_length > 0:
            pad_param = [[0, 0] for _ in mel_spec.shape]
            pad_param[-1][1] = self.segment_length - last_segment_length
            padded_segment = np.pad(
                mel_spec[..., full_segments * self.segment_length:], pad_param
            )
            segments.append(padded_segment)

        batch = []
        for i, seg in enumerate(segments):
            batch.append(
                {
                    "dataset": example["dataset"],
                    "example_id": f"{example['example_id']}_s{i}",
                    "mel_spec": seg,
                    "seq_len": 1,  # dummy
                    "weak_targets": np.zeros(1),  # dummy
                    "strong_targets": np.zeros(1),  # dummy
                    "boundary_targets": np.zeros(1),  # dummy
                }
            )
        return batch


@dataclasses.dataclass
class Transform:
    stft: STFT
    mel_transform: MelTransform
    cropping: RandomTimeCrop

    def __call__(self, example):
        audio = example["audio_data"]
        assert len(audio.shape) == 2, (audio.shape, "channel, time dimension")
        stft_signal = self.stft(audio)
        assert len(stft_signal.shape) == 3, stft_signal.shape
        spec = stft_to_spectrogram(stft_signal)
        mel_spec = self.mel_transform(spec)
        mel_spec = np.asarray(mel_spec).swapaxes(-1, -2)  # (..., mel, time)
        if self.cropping:
            mel_spec = self.cropping(mel_spec)
        # Normalized using input layer in model

        mel_spec = mel_spec.astype(np.float32)

        # TODO: augmentation functions
        # tfms: augmentation functions
        #     # Apply transforms
        #     if self.tfms is not None:
        #         if not dont_tfms:
        #             lms = self.tfms(lms)
        #####################
        return {
            "dataset": example["dataset"],
            "example_id": example["example_id"],
            "mel_spec": mel_spec,
            "seq_len": 1,  # dummy
            "weak_targets": np.zeros(1),  # dummy
            "strong_targets": np.zeros(1),  # dummy
            "boundary_targets": np.zeros(1),  # dummy
        }

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config["stft"] = {
            "factory": "paderbox.transform.module_stft.STFT",
            "size": 400,
            "shift": 160,
        }
        config["mel_transform"] = {
            "factory": "paderbox.transform.module_fbank.MelTransform",
            "sample_rate": 16_000,
            "stft_size": 400,
            "number_of_filters": 80,
            "lowest_frequency": 50,
            "highest_frequency": 8000,
        }
        config["cropping"] = {
            "factory": "msm_mae.padertorch.transform.RandomTimeCrop"}


@dataclasses.dataclass
class OrigTransform(nn.Module):
    sample_rate = 16000
    window_size = 400
    n_fft = 400
    hop_size = 160
    n_mels = 80
    f_min = 50
    f_max = 8000
    norm_stats = (-6.0385, 4.0184)

    def __post_init__(self):
        super().__init__()
        self.to_spec = nnAudio.Spectrogram.MelSpectrogram(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.window_size,
            hop_length=self.hop_size,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            center=True,
            power=2,
            verbose=False,
        )

    def __call__(self, example):
        audio = example["audio_data"]
        audio = audio.astype(np.float32)
        x = self.to_spec(torch.tensor(audio))

        mel_spec = (x + torch.finfo().eps).log()
        if self.norm_stats is not None:
            mel_spec = (mel_spec - self.norm_stats[0]) / self.norm_stats[1]
        mel_spec = np.asarray(mel_spec)
        return {
            "dataset": example["dataset"],
            "example_id": example["example_id"],
            "mel_spec": mel_spec,
            "seq_len": 1,  # dummy
            "weak_targets": np.zeros(1),  # dummy
            "strong_targets": np.zeros(1),  # dummy
            "boundary_targets": np.zeros(1),  # dummy
        }
