from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from fireredasr2s.fireredlid.lid import FireRedLid, FireRedLidConfig
from fireredasr2s.fireredlid.runtime.batch_planner import FeatureItem


class FakeTokenizer:
    def detokenize(self, ids):
        return str(ids[0])


class FakeModel:
    def __init__(self):
        self.batch_sizes = []

    def process(self, features, lengths, *args, stage_recorder=None):
        self.batch_sizes.append(features.size(0))
        return [
            [
                {
                    "yseq": torch.tensor(
                        [int(features[index, 0, 0].item())]
                    ),
                    "confidence": torch.tensor(0.9),
                }
            ]
            for index in range(features.size(0))
        ]


class RecordingStageRecorder:
    def __init__(self):
        self.names = []

    @contextmanager
    def measure(self, name):
        self.names.append(name)
        yield


def item(index, value, duration):
    return FeatureItem(
        index=index,
        uttid=f"utt-{index}",
        wav_input=f"{index}.wav",
        feature=torch.full(
            (int(duration * 10), 80),
            float(value),
        ),
        duration_s=duration,
        processed_duration_s=duration,
        truncated=False,
    )


def make_lid(config):
    lid = FireRedLid.__new__(FireRedLid)
    lid.model = FakeModel()
    lid.tokenizer = FakeTokenizer()
    lid.config = config
    lid.backend_max_batch = None
    lid.active_backend = config.backend
    return lid


def test_config_resolves_profile_defaults():
    assert FireRedLidConfig(
        profile="latency"
    ).resolved_batch_strategy == "none"
    assert FireRedLidConfig(
        profile="throughput"
    ).resolved_batch_strategy == "auto"


def test_tensorrt_requires_gpu_half_and_engine_dir():
    with pytest.raises(ValueError, match="use_half=True"):
        FireRedLidConfig(backend="tensorrt")


def test_only_eager_is_accepted_as_explicit_fallback():
    with pytest.raises(ValueError, match="fallback_backend"):
        FireRedLidConfig(
            backend="compile",
            fallback_backend="compile",
        )


def test_backend_initialization_falls_back_only_when_requested(monkeypatch):
    class BrokenCompileBackend:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("compile setup failed")

    monkeypatch.setattr(
        "fireredasr2s.fireredlid.lid.CompileEncoderBackend",
        BrokenCompileBackend,
    )
    strict = FireRedLid.__new__(FireRedLid)
    strict.model = SimpleNamespace(encoder=torch.nn.Identity())
    strict.config = FireRedLidConfig(use_gpu=False, backend="compile")
    with pytest.raises(
        RuntimeError,
        match="failed to initialize compile backend",
    ):
        strict._configure_encoder_backend()

    fallback = FireRedLid.__new__(FireRedLid)
    fallback.model = SimpleNamespace(encoder=torch.nn.Identity())
    fallback.config = FireRedLidConfig(
        use_gpu=False,
        backend="compile",
        fallback_backend="eager",
    )
    fallback._configure_encoder_backend()
    assert fallback.active_backend == "eager"
    assert isinstance(fallback.model.encoder, torch.nn.Identity)


def test_infer_items_restores_original_order_after_bucketing():
    lid = make_lid(
        FireRedLidConfig(
            use_gpu=False,
            backend="eager",
            profile="throughput",
            batch_strategy="bucket",
            max_sub_batch_size=2,
        )
    )

    results = lid._infer_items(
        [
            item(0, 30, 30),
            item(1, 4, 4),
            item(2, 10, 10),
        ]
    )

    assert [result["uttid"] for result in results] == [
        "utt-0",
        "utt-1",
        "utt-2",
    ]
    assert [result["lang"] for result in results] == ["30", "4", "10"]


def test_engine_max_batch_is_a_hard_upper_bound():
    lid = make_lid(
        FireRedLidConfig(
            use_gpu=False,
            backend="eager",
            batch_strategy="none",
            max_sub_batch_size=4,
        )
    )
    lid.backend_max_batch = 1

    lid._infer_items(
        [
            item(0, 1, 1),
            item(1, 2, 1),
            item(2, 3, 1),
        ]
    )

    assert lid.model.batch_sizes == [1, 1, 1]


def test_infer_items_records_transfer_and_result_formatting_stages():
    lid = make_lid(FireRedLidConfig(use_gpu=False))
    recorder = RecordingStageRecorder()
    lid.stage_recorder = recorder

    lid._infer_items([item(0, 1, 1)])

    assert "h2d" in recorder.names
    assert "result_formatting" in recorder.names


def test_process_records_fbank_stage_for_empty_features():
    class EmptyFeatureExtractor:
        def extract_many(self, *args, **kwargs):
            return []

    lid = FireRedLid.__new__(FireRedLid)
    lid.config = FireRedLidConfig(use_gpu=False)
    lid.feat_extractor = EmptyFeatureExtractor()
    recorder = RecordingStageRecorder()
    lid.stage_recorder = recorder

    result = lid.process(["empty"], ["empty.wav"])

    assert result == [{"uttid": "empty", "lang": ""}]
    assert recorder.names == ["fbank"]
