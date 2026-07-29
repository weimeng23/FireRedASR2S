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


class TinyPrecisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.lid_decoder = torch.nn.Linear(2, 2)


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
    lid.encoder_dtype = torch.float32
    lid.backend_max_batch = None
    lid.active_backend = config.backend
    return lid


def test_runtime_config_does_not_expose_batch_scheduling_options():
    config = FireRedLidConfig()

    assert not hasattr(config, "batch_strategy")
    assert not hasattr(config, "max_sub_batch_size")
    assert not hasattr(config, "resolved_batch_strategy")


def test_model_api_preserves_fp32_defaults():
    config = FireRedLidConfig()

    assert config.encoder_precision == "fp32"
    assert config.decoder_precision == "fp32"


def test_config_supports_independent_encoder_and_decoder_precision():
    config = FireRedLidConfig(
        encoder_precision="bf16",
        decoder_precision="fp32",
    )

    assert config.encoder_precision == "bf16"
    assert config.decoder_precision == "fp32"


def test_legacy_use_half_rejects_explicit_model_precision():
    with pytest.raises(ValueError, match="use_half.*precision"):
        FireRedLidConfig(
            use_half=True,
            encoder_precision="fp16",
            decoder_precision="fp32",
        )


def test_runtime_casts_encoder_and_decoder_independently():
    model = TinyPrecisionModel()

    FireRedLid(
        feat_extractor=object(),
        model=model,
        tokenizer=object(),
        config=FireRedLidConfig(
            use_gpu=False,
            encoder_precision="bf16",
            decoder_precision="fp32",
        ),
    )

    assert model.encoder.weight.dtype == torch.bfloat16
    assert model.lid_decoder.weight.dtype == torch.float32


def test_runtime_rejects_bf16_when_cuda_device_does_not_support_it(
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(RuntimeError, match="BF16"):
        FireRedLid(
            feat_extractor=object(),
            model=TinyPrecisionModel(),
            tokenizer=object(),
            config=FireRedLidConfig(
                use_gpu=True,
                encoder_precision="bf16",
                decoder_precision="fp32",
            ),
        )


def test_tensorrt_requires_gpu_fp16_encoder_and_engine_dir():
    with pytest.raises(ValueError, match="encoder_precision='fp16'"):
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


def test_infer_items_executes_one_physical_batch_in_input_order():
    lid = make_lid(
        FireRedLidConfig(
            use_gpu=False,
            backend="eager",
            profile="throughput",
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
    assert lid.model.batch_sizes == [3]


def test_engine_rejects_batch_over_backend_limit_without_splitting():
    lid = make_lid(
        FireRedLidConfig(
            use_gpu=False,
            backend="eager",
        )
    )
    lid.backend_max_batch = 1

    with pytest.raises(ValueError, match="backend maximum batch size"):
        lid._infer_items(
            [
                item(0, 1, 1),
                item(1, 2, 1),
                item(2, 3, 1),
            ]
        )

    assert lid.model.batch_sizes == []


def test_infer_items_records_transfer_and_result_formatting_stages():
    lid = make_lid(FireRedLidConfig(use_gpu=False))
    recorder = RecordingStageRecorder()
    lid.stage_recorder = recorder

    lid._infer_items([item(0, 1, 1)])

    assert "h2d" in recorder.names
    assert "result_formatting" in recorder.names


def test_process_returns_no_results_for_empty_features():
    class EmptyFeatureExtractor:
        def extract_many(self, *args, **kwargs):
            return []

    lid = FireRedLid.__new__(FireRedLid)
    lid.config = FireRedLidConfig(use_gpu=False)
    lid.feat_extractor = EmptyFeatureExtractor()
    recorder = RecordingStageRecorder()
    lid.stage_recorder = recorder

    result = lid.process(["empty"], ["empty.wav"])

    assert result == []
    assert recorder.names == ["fbank"]


def test_process_preserves_model_inference_error():
    class OneFeatureExtractor:
        def extract_many(self, *args, **kwargs):
            return [item(0, 1, 1)]

    class BrokenModel(FakeModel):
        def process(self, *args, **kwargs):
            raise RuntimeError("decoder produced NaN")

    lid = make_lid(FireRedLidConfig(use_gpu=False))
    lid.feat_extractor = OneFeatureExtractor()
    lid.model = BrokenModel()

    with pytest.raises(RuntimeError, match="decoder produced NaN"):
        lid.process(["broken"], ["broken.wav"])
