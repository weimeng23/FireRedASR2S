import numpy as np
import torch

from fireredasr2s.fireredlid.data.feat import (
    FeatExtractor,
    KaldifeatFbank,
)


class FakeFbank:
    def __init__(self):
        self.seen_samples = []

    def __call__(self, wav):
        sample_rate, samples = wav
        self.seen_samples.append((sample_rate, samples.copy()))
        return np.tile(
            np.arange(80, dtype=np.float32),
            (len(samples), 1),
        )


def make_extractor():
    extractor = FeatExtractor.__new__(FeatExtractor)
    extractor.fbank = FakeFbank()
    extractor.cmvn = lambda value: value + 10.0
    return extractor


def test_extract_many_truncates_before_fbank_and_keeps_original_duration():
    extractor = make_extractor()
    samples = np.arange(30, dtype=np.int16)

    items = extractor.extract_many(
        [(10, samples)],
        ["utt"],
        max_audio_seconds=2.0,
    )

    assert len(extractor.fbank.seen_samples[0][1]) == 20
    assert items[0].duration_s == 3.0
    assert items[0].processed_duration_s == 2.0
    assert items[0].truncated is True
    assert torch.equal(
        items[0].feature[0],
        torch.arange(80).float() + 10.0,
    )


def test_legacy_call_applies_cmvn_before_zero_padding():
    extractor = make_extractor()
    first = np.arange(20, dtype=np.int16)
    second = np.arange(10, dtype=np.int16)

    padded, lengths, durations, wavs, uttids = extractor(
        [(10, first), (10, second)],
        ["first", "second"],
    )

    assert lengths.tolist() == [20, 10]
    assert torch.count_nonzero(padded[1, 10:]) == 0
    assert durations == [2.0, 1.0]
    assert uttids == ["first", "second"]
    assert len(wavs) == 2


def test_kaldifeat_fbank_applies_configured_frame_geometry():
    fbank = KaldifeatFbank(frame_length=30, frame_shift=12)

    assert fbank.opts.frame_opts.frame_length_ms == 30
    assert fbank.opts.frame_opts.frame_shift_ms == 12
