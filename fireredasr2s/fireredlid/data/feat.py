# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Yan Jia)

import math
import os

import kaldiio
import kaldi_native_fbank as knf
import numpy as np
import torch

from ..runtime.batch_planner import FeatureItem, pad_features


class FeatExtractor:
    def __init__(self, kaldi_cmvn_file):
        self.cmvn = CMVN(kaldi_cmvn_file) if kaldi_cmvn_file != "" else None
        self.fbank = KaldifeatFbank(num_mel_bins=80, frame_length=25,
            frame_shift=10, dither=0.0)

    def _load_waveform(self, wav_input):
        if isinstance(wav_input, str):
            return kaldiio.load_mat(wav_input)
        sample_rate, wav_np = wav_input
        return sample_rate, wav_np

    def extract_many(
        self, wav_inputs, wav_uttids, max_audio_seconds=None
    ):
        items = []
        for index, (wav_input, uttid) in enumerate(
            zip(wav_inputs, wav_uttids)
        ):
            sample_rate, wav_np = self._load_waveform(wav_input)
            duration_s = wav_np.shape[0] / sample_rate
            max_samples = wav_np.shape[0]
            if max_audio_seconds is not None:
                max_samples = min(
                    max_samples, int(sample_rate * max_audio_seconds)
                )
            processed_wav = wav_np[:max_samples]
            processed_duration_s = processed_wav.shape[0] / sample_rate
            fbank = self.fbank((sample_rate, processed_wav))
            if fbank.shape[0] < 1:
                continue
            if self.cmvn is not None:
                fbank = self.cmvn(fbank)
            items.append(
                FeatureItem(
                    index=index,
                    uttid=uttid,
                    wav_input=wav_input,
                    feature=torch.from_numpy(fbank).float(),
                    duration_s=duration_s,
                    processed_duration_s=processed_duration_s,
                    truncated=processed_wav.shape[0] != wav_np.shape[0],
                )
            )
        return items

    def __call__(self, wav_paths, wav_uttids):
        items = self.extract_many(wav_paths, wav_uttids)
        if not items:
            return None, None, [], [], []
        features = pad_features([item.feature for item in items])
        lengths = torch.tensor(
            [item.feature.size(0) for item in items], dtype=torch.long
        )
        return (
            features,
            lengths,
            [item.duration_s for item in items],
            [item.wav_input for item in items],
            [item.uttid for item in items],
        )

    def pad_feat(self, xs, pad_value):
        return pad_features(xs, pad_value)


class CMVN:
    def __init__(self, kaldi_cmvn_file):
        self.dim, self.means, self.inverse_std_variences = \
            self.read_kaldi_cmvn(kaldi_cmvn_file)

    def __call__(self, x, is_train=False):
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        out = x - self.means
        out = out * self.inverse_std_variences
        return out

    def read_kaldi_cmvn(self, kaldi_cmvn_file):
        assert os.path.exists(kaldi_cmvn_file)
        stats = kaldiio.load_mat(kaldi_cmvn_file)
        assert stats.shape[0] == 2
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        assert count >= 1
        floor = 1e-20
        means = []
        inverse_std_variences = []
        for d in range(dim):
            mean = stats[0, d] / count
            means.append(mean.item())
            varience = (stats[1, d] / count) - mean*mean
            if varience < floor:
                varience = floor
            istd = 1.0 / math.sqrt(varience)
            inverse_std_variences.append(istd)
        return dim, np.array(means), np.array(inverse_std_variences)



class KaldifeatFbank:
    def __init__(self, num_mel_bins=80, frame_length=25, frame_shift=10,
                 dither=1.0):
        self.dither = dither
        opts = knf.FbankOptions()
        opts.frame_opts.dither = dither
        opts.mel_opts.num_bins = num_mel_bins
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts

    def __call__(self, wav, is_train=False):
        if type(wav) is str:
            sample_rate, wav_np = kaldiio.load_mat(wav)
        elif type(wav) in [tuple, list] and len(wav) == 2:
            sample_rate, wav_np = wav
        assert len(wav_np.shape) == 1

        dither = self.dither if is_train else 0.0
        self.opts.frame_opts.dither = dither
        fbank = knf.OnlineFbank(self.opts)

        fbank.accept_waveform(sample_rate, wav_np.tolist())
        feat = []
        for i in range(fbank.num_frames_ready):
            feat.append(fbank.get_frame(i))
        if len(feat) == 0:
            print("Check data, len(feat) == 0", wav, flush=True)
            return np.zeros((0, self.opts.mel_opts.num_bins))
        feat = np.vstack(feat)
        return feat
