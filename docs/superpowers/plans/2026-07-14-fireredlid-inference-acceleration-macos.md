# FireRedLID macOS Inference Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the macOS portion of a backward-compatible FireRedLID runtime with eager, compile, and TensorRT Encoder backend boundaries, dynamic ONNX export, optional logical-batch planning, and reproducible verification/benchmark tools.

**Architecture:** Keep the official `FireRedLid` API, FBank/CMVN math, Transformer Decoder, and beam search. Split per-utterance feature extraction from physical-batch padding, route padded features through a common Encoder backend contract, and preserve the official three-value Encoder interface natively in every backend. macOS validates eager behavior and dynamic FP32 ONNX; Linux GPU engine construction and CUDA performance acceptance are explicitly outside this plan.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, kaldiio, kaldi-native-fbank, ONNX opset 17, ONNX Runtime CPU, PyYAML, pytest.

## Global Constraints

- Preserve `FireRedLID/model.pth.tar`, `cmvn.ark`, `dict.txt`, official Decoder weights, `beam_size=3`, `decode_max_len=2`, and confidence calculation.
- Keep `backend="eager"` as the default and preserve the existing `FireRedLid.from_pretrained(...).process(...)` entry point.
- Truncate waveforms to at most `60.0` seconds before FBank; keep original `dur_s` and expose processed duration only in diagnostics.
- Apply CMVN to each valid `[T_i, 80]` feature before padding; padding remains zero.
- Do not add request queues, Triton serving, TensorRT-LLM, FBank worker pools, feature caches, or INT8/FP8.
- Eager, compile, ONNX, and TensorRT Encoder boundaries all return `encoder_outputs`, `encoder_lengths`, and `encoder_mask`.
- Explicit `compile` or `tensorrt` backend initialization fails clearly unless `fallback_backend="eager"` is set.
- No claim that TensorRT or CUDA `torch.compile` works until executed in a Linux NVIDIA environment.
- Treat `/Users/weimeng/workspace/FireRedASR2S/FireRedLID/` as local untracked model data; never stage it.

## Scope Boundary

This plan implements design Phase A on macOS. It writes the Linux TensorRT builder and runtime surface so they can be inspected and tested for lazy imports, manifests, and shape validation, but it does not mark engine deserialization, CUDA execution, or GPU benchmarks as passing. Once RTX PRO 5000 or L20 access exists, create a separate Linux execution plan using the artifacts produced here.

## File Map

**Modify**

- `.gitignore`: keep generated runtime artifacts out of commits.
- `pyproject.toml`: declare the missing official FBank dependency.
- `requirements.txt`: replace the unavailable PyPI 1.15 pin with the exact official v1.15 commit.
- `fireredasr2s/fireredlid/data/feat.py`: expose per-utterance extraction and 60-second truncation while preserving the legacy call.
- `fireredasr2s/fireredlid/lid.py`: extend config, select backends, execute planned physical batches, and preserve result ordering.
- `fireredasr2s/fireredlid/models/fireredlid_aed.py`: add opt-in benchmark timing hooks around Encoder and Decoder without changing default inference.
- `fireredasr2s/fireredlid/models/module/conformer_encoder.py`: vectorize mask creation and remove the unused layer-output list.

**Create package modules**

- `fireredasr2s/fireredlid/runtime/__init__.py`: public runtime exports.
- `fireredasr2s/fireredlid/runtime/batch_planner.py`: feature item types, padding, none/bucket/auto planning.
- `fireredasr2s/fireredlid/runtime/encoder_backend.py`: result type, abstract backend, compatibility adapter.
- `fireredasr2s/fireredlid/runtime/pytorch_backend.py`: eager and compile Encoder implementations.
- `fireredasr2s/fireredlid/runtime/tensorrt_backend.py`: manifest validation, lazy TensorRT import, engine execution surface.

**Create tools**

- `runtime/fireredlid/pyproject.toml`: isolated Mac export dependencies.
- `runtime/fireredlid/profiles.yaml`: initial dynamic-shape profile.
- `runtime/fireredlid/export_encoder_onnx.py`: FP32 dynamic ONNX export.
- `runtime/fireredlid/build_engine.py`: Linux FP16 TensorRT builder and manifest writer.
- `runtime/fireredlid/verify.py`: eager/ONNX numerical and label verification.
- `runtime/fireredlid/benchmark.py`: latency/throughput metrics and JSON output.
- `runtime/fireredlid/example_manifest.jsonl`: reproducible two-file smoke/benchmark input.
- `runtime/fireredlid/README.md`: Mac commands, Linux handoff, and limitations.

**Create tests**

- `tests/fireredlid/test_conformer_encoder.py`
- `tests/fireredlid/test_batch_planner.py`
- `tests/fireredlid/test_feat.py`
- `tests/fireredlid/test_encoder_backend.py`
- `tests/fireredlid/test_lid_runtime.py`
- `tests/fireredlid/test_onnx_export.py`
- `tests/fireredlid/test_tensorrt_backend.py`
- `tests/fireredlid/test_benchmark.py`

---

### Task 1: Make the Conformer Encoder mask exportable

**Files:**
- Modify: `fireredasr2s/fireredlid/models/module/conformer_encoder.py:26-51`
- Test: `tests/fireredlid/test_conformer_encoder.py`

**Interfaces:**
- Consumes: `ConformerEncoder.padding_position_is_0(padded_input, input_lengths)`.
- Produces: the same `[B, 1, T]` `torch.uint8` mask without Tensor-indexed Python slicing.

- [ ] **Step 1: Write the exportability and exact-mask tests**

```python
import torch

from fireredasr2s.fireredlid.models.module.conformer_encoder import ConformerEncoder


class MaskModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConformerEncoder(
            idim=80,
            n_layers=0,
            n_head=2,
            d_model=8,
            residual_dropout=0.0,
            dropout_rate=0.0,
            kernel_size=3,
            pe_maxlen=64,
        )

    def forward(self, features, lengths):
        return self.encoder.padding_position_is_0(features, lengths)


def test_padding_mask_is_exact():
    module = MaskModule()
    features = torch.zeros(2, 5, 80)
    lengths = torch.tensor([3, 5])
    actual = module(features, lengths)
    expected = torch.tensor([
        [[1, 1, 1, 0, 0]],
        [[1, 1, 1, 1, 1]],
    ], dtype=torch.uint8)
    assert torch.equal(actual, expected)


def test_padding_mask_is_torch_exportable():
    module = MaskModule().eval()
    features = torch.zeros(2, 5, 80)
    lengths = torch.tensor([3, 5])
    exported = torch.export.export(module, (features, lengths))
    assert torch.equal(
        exported.module()(features, lengths),
        module(features, lengths),
    )
```

- [ ] **Step 2: Run the test and confirm the current loop is not export-safe**

Run: `python3 -m pytest tests/fireredlid/test_conformer_encoder.py -v`

Expected: the exact-value test passes and the export test fails at the Tensor-valued slice assignment in `padding_position_is_0()`.

- [ ] **Step 3: Vectorize the mask and remove the unused layer-output list**

```python
def padding_position_is_0(self, padded_input, input_lengths):
    time_index = torch.arange(
        padded_input.size(1),
        device=padded_input.device,
    )
    mask = time_index.unsqueeze(0) < input_lengths.unsqueeze(1)
    return mask.unsqueeze(1).to(torch.uint8)
```

Replace the Encoder layer loop with:

```python
for enc_layer in self.layer_stack:
    enc_output = enc_layer(
        enc_output,
        pos_emb,
        slf_attn_mask=src_mask,
        pad_mask=src_mask,
    )
```

- [ ] **Step 4: Run the focused and syntax tests**

Run: `python3 -m pytest tests/fireredlid/test_conformer_encoder.py -v`

Expected: `2 passed`.

Run: `python3 -m compileall -q fireredasr2s/fireredlid/models/module/conformer_encoder.py`

Expected: exit code `0`.

- [ ] **Step 5: Commit**

```bash
git add fireredasr2s/fireredlid/models/module/conformer_encoder.py tests/fireredlid/test_conformer_encoder.py
git commit -m "perf(fireredlid): vectorize encoder padding mask"
```

---

### Task 2: Add deterministic physical-batch planning

**Files:**
- Create: `fireredasr2s/fireredlid/runtime/__init__.py`
- Create: `fireredasr2s/fireredlid/runtime/batch_planner.py`
- Test: `tests/fireredlid/test_batch_planner.py`

**Interfaces:**
- Produces: `FeatureItem`, `PlannedBatch`, `BatchPlanner.plan(items)`, and `pad_features(features)`.
- Consumers: Task 3 feature extraction and Task 5 inference orchestration.

- [ ] **Step 1: Write tests for none, bucket, auto, padding, and original indices**

```python
import torch

from fireredasr2s.fireredlid.runtime.batch_planner import (
    BatchPlanner,
    FeatureItem,
)


def make_item(index: int, frames: int, duration_s: float) -> FeatureItem:
    return FeatureItem(
        index=index,
        uttid=f"utt-{index}",
        wav_input=f"{index}.wav",
        feature=torch.full((frames, 80), float(index + 1)),
        duration_s=duration_s,
        processed_duration_s=duration_s,
        truncated=False,
    )


def batch_indices(plans):
    return [[item.index for item in plan.items] for plan in plans]


def test_none_preserves_order_and_chunks():
    items = [make_item(0, 4, 1), make_item(1, 2, 1), make_item(2, 3, 1)]
    plans = BatchPlanner("none", max_sub_batch_size=2).plan(items)
    assert batch_indices(plans) == [[0, 1], [2]]
    assert plans[0].padded_features.shape == (2, 4, 80)
    assert plans[0].feature_lengths.tolist() == [4, 2]


def test_bucket_groups_by_processed_duration():
    items = [make_item(0, 3000, 30), make_item(1, 400, 4), make_item(2, 1000, 10)]
    plans = BatchPlanner("bucket", max_sub_batch_size=8).plan(items)
    assert batch_indices(plans) == [[1], [2], [0]]


def test_auto_buckets_only_when_saving_reaches_twenty_percent():
    mixed = [make_item(0, 6000, 60), make_item(1, 500, 5), make_item(2, 500, 5)]
    close = [make_item(0, 5000, 50), make_item(1, 5500, 55)]
    planner = BatchPlanner("auto", max_sub_batch_size=8)
    assert batch_indices(planner.plan(mixed)) == [[1, 2], [0]]
    assert batch_indices(planner.plan(close)) == [[0, 1]]
```

- [ ] **Step 2: Run the tests and confirm the module is missing**

Run: `python3 -m pytest tests/fireredlid/test_batch_planner.py -v`

Expected: collection fails with `ModuleNotFoundError: fireredasr2s.fireredlid.runtime`.

- [ ] **Step 3: Implement the planner and data types**

```python
from bisect import bisect_left
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class FeatureItem:
    index: int
    uttid: str
    wav_input: object
    feature: Tensor
    duration_s: float
    processed_duration_s: float
    truncated: bool


@dataclass(frozen=True)
class PlannedBatch:
    items: tuple[FeatureItem, ...]
    padded_features: Tensor
    feature_lengths: Tensor


def pad_features(features: Sequence[Tensor], pad_value: float = 0.0) -> Tensor:
    if not features:
        raise ValueError("features must not be empty")
    max_frames = max(feature.size(0) for feature in features)
    padded = features[0].new_full((len(features), max_frames, features[0].size(1)), pad_value)
    for index, feature in enumerate(features):
        padded[index, : feature.size(0)] = feature
    return padded


class BatchPlanner:
    VALID_STRATEGIES = {"none", "bucket", "auto"}

    def __init__(
        self,
        strategy: str,
        max_sub_batch_size: int | None,
        bucket_boundaries_s: tuple[float, ...] = (5.0, 15.0, 30.0, 60.0),
        auto_min_saving_ratio: float = 0.20,
    ):
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"unsupported batch strategy: {strategy}")
        if max_sub_batch_size is not None and max_sub_batch_size < 1:
            raise ValueError("max_sub_batch_size must be positive")
        self.strategy = strategy
        self.max_sub_batch_size = max_sub_batch_size
        self.bucket_boundaries_s = bucket_boundaries_s
        self.auto_min_saving_ratio = auto_min_saving_ratio

    def _chunk(self, items: Sequence[FeatureItem]) -> list[list[FeatureItem]]:
        size = self.max_sub_batch_size or max(1, len(items))
        return [list(items[start : start + size]) for start in range(0, len(items), size)]

    def _sequential_groups(self, items: Sequence[FeatureItem]) -> list[list[FeatureItem]]:
        return self._chunk(items)

    def _bucket_groups(self, items: Sequence[FeatureItem]) -> list[list[FeatureItem]]:
        buckets: dict[int, list[FeatureItem]] = {}
        for item in items:
            key = bisect_left(self.bucket_boundaries_s, item.processed_duration_s)
            buckets.setdefault(key, []).append(item)
        groups: list[list[FeatureItem]] = []
        for key in sorted(buckets):
            groups.extend(self._chunk(buckets[key]))
        return groups

    @staticmethod
    def _padded_frames(groups: Sequence[Sequence[FeatureItem]]) -> int:
        return sum(len(group) * max(item.feature.size(0) for item in group) for group in groups)

    def plan(self, items: Sequence[FeatureItem]) -> list[PlannedBatch]:
        if not items:
            return []
        direct = self._sequential_groups(items)
        bucketed = self._bucket_groups(items)
        groups = direct
        if self.strategy == "bucket":
            groups = bucketed
        elif self.strategy == "auto":
            saving = 1.0 - self._padded_frames(bucketed) / self._padded_frames(direct)
            if saving >= self.auto_min_saving_ratio:
                groups = bucketed
        return [
            PlannedBatch(
                items=tuple(group),
                padded_features=pad_features([item.feature for item in group]),
                feature_lengths=torch.tensor([item.feature.size(0) for item in group], dtype=torch.long),
            )
            for group in groups
        ]
```

Export the four public names from `runtime/__init__.py`.

- [ ] **Step 4: Run planner tests**

Run: `python3 -m pytest tests/fireredlid/test_batch_planner.py -v`

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add fireredasr2s/fireredlid/runtime tests/fireredlid/test_batch_planner.py
git commit -m "feat(fireredlid): add physical batch planner"
```

---

### Task 3: Separate per-utterance FBank/CMVN from padding

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `fireredasr2s/fireredlid/data/feat.py:12-58`
- Test: `tests/fireredlid/test_feat.py`

**Interfaces:**
- Consumes: `FeatureItem` and `pad_features` from Task 2.
- Produces: `FeatExtractor.extract_many(wav_inputs, uttids, max_audio_seconds)` while preserving the legacy `FeatExtractor.__call__()` return tuple.

- [ ] **Step 1: Declare and install the official missing dependency**

Add this dependency to `pyproject.toml` next to `kaldiio`:

```toml
"kaldi-native-fbank @ git+https://github.com/csukuangfj/kaldi-native-fbank.git@f68c6b43f739697d7ab02ff6debacee130e1d541",
```

Use the same direct reference in `requirements.txt`, then run:

`python3 -m pip install "git+https://github.com/csukuangfj/kaldi-native-fbank.git@f68c6b43f739697d7ab02ff6debacee130e1d541"`

Expected: installation succeeds and `python3 -c "import kaldi_native_fbank"` exits `0`.

- [ ] **Step 2: Write truncation, CMVN-order, and legacy-call tests**

```python
import numpy as np
import torch

from fireredasr2s.fireredlid.data.feat import FeatExtractor


class FakeFbank:
    def __init__(self):
        self.seen_samples = []

    def __call__(self, wav):
        sample_rate, samples = wav
        self.seen_samples.append((sample_rate, samples.copy()))
        return np.tile(np.arange(80, dtype=np.float32), (len(samples), 1))


def make_extractor():
    extractor = FeatExtractor.__new__(FeatExtractor)
    extractor.fbank = FakeFbank()
    extractor.cmvn = lambda value: value + 10.0
    return extractor


def test_extract_many_truncates_before_fbank_and_keeps_original_duration():
    extractor = make_extractor()
    samples = np.arange(30, dtype=np.int16)
    items = extractor.extract_many([(10, samples)], ["utt"], max_audio_seconds=2.0)
    assert len(extractor.fbank.seen_samples[0][1]) == 20
    assert items[0].duration_s == 3.0
    assert items[0].processed_duration_s == 2.0
    assert items[0].truncated is True
    assert torch.equal(items[0].feature[0], torch.arange(80).float() + 10.0)


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
```

- [ ] **Step 3: Run tests and verify the new method is missing**

Run: `python3 -m pytest tests/fireredlid/test_feat.py -v`

Expected: FAIL with `AttributeError: 'FeatExtractor' object has no attribute 'extract_many'`.

- [ ] **Step 4: Implement per-item extraction and compose the legacy call**

```python
def _load_waveform(self, wav_input):
    if isinstance(wav_input, str):
        return kaldiio.load_mat(wav_input)
    sample_rate, wav_np = wav_input
    return sample_rate, wav_np


def extract_many(self, wav_inputs, wav_uttids, max_audio_seconds=None):
    items = []
    for index, (wav_input, uttid) in enumerate(zip(wav_inputs, wav_uttids)):
        sample_rate, wav_np = self._load_waveform(wav_input)
        duration_s = wav_np.shape[0] / sample_rate
        max_samples = wav_np.shape[0]
        if max_audio_seconds is not None:
            max_samples = min(max_samples, int(sample_rate * max_audio_seconds))
        processed_wav = wav_np[:max_samples]
        processed_duration_s = processed_wav.shape[0] / sample_rate
        fbank = self.fbank((sample_rate, processed_wav))
        if fbank.shape[0] < 1:
            continue
        if self.cmvn is not None:
            fbank = self.cmvn(fbank)
        items.append(FeatureItem(
            index=index,
            uttid=uttid,
            wav_input=wav_input,
            feature=torch.from_numpy(fbank).float(),
            duration_s=duration_s,
            processed_duration_s=processed_duration_s,
            truncated=processed_wav.shape[0] != wav_np.shape[0],
        ))
    return items


def __call__(self, wav_paths, wav_uttids):
    items = self.extract_many(wav_paths, wav_uttids)
    if not items:
        return None, None, [], [], []
    features = pad_features([item.feature for item in items])
    lengths = torch.tensor([item.feature.size(0) for item in items], dtype=torch.long)
    return (
        features,
        lengths,
        [item.duration_s for item in items],
        [item.wav_input for item in items],
        [item.uttid for item in items],
    )
```

Import `FeatureItem` and `pad_features` from `..runtime.batch_planner`. Do not change `KaldifeatFbank` or `CMVN` math.

- [ ] **Step 5: Run focused and existing import checks**

Run: `python3 -m pytest tests/fireredlid/test_feat.py tests/fireredlid/test_batch_planner.py -v`

Expected: `5 passed`.

Run: `python3 -c "from fireredasr2s.fireredlid.data.feat import FeatExtractor"`

Expected: exit code `0`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt fireredasr2s/fireredlid/data/feat.py tests/fireredlid/test_feat.py docs/superpowers/plans/2026-07-14-fireredlid-inference-acceleration-macos.md
git commit -m "refactor(fireredlid): separate feature extraction from padding"
```

---

### Task 4: Define eager and compile Encoder backends

**Files:**
- Create: `fireredasr2s/fireredlid/runtime/encoder_backend.py`
- Create: `fireredasr2s/fireredlid/runtime/pytorch_backend.py`
- Modify: `fireredasr2s/fireredlid/runtime/__init__.py`
- Test: `tests/fireredlid/test_encoder_backend.py`

**Interfaces:**
- Produces: `EncoderResult(outputs, lengths, mask)`, `EncoderBackend.encode(...)`, `CompatibleEncoderAdapter.forward(...)`, `EagerEncoderBackend`, and `CompileEncoderBackend`.
- Consumers: Task 5 runtime integration and Task 7 TensorRT backend.

- [ ] **Step 1: Write adapter and backend tests with a fake official Encoder**

```python
import torch

from fireredasr2s.fireredlid.runtime.encoder_backend import CompatibleEncoderAdapter
from fireredasr2s.fireredlid.runtime.pytorch_backend import (
    CompileEncoderBackend,
    EagerEncoderBackend,
)


class FakeEncoder(torch.nn.Module):
    def forward(self, features, lengths):
        mask = (
            torch.arange(features.size(1))[None, :] < lengths[:, None]
        ).unsqueeze(1).to(torch.uint8)
        return features + 1.0, mask.sum(-1).squeeze(1), mask


def test_adapter_preserves_official_three_output_contract():
    adapter = CompatibleEncoderAdapter(EagerEncoderBackend(FakeEncoder()))
    features = torch.zeros(2, 4, 3)
    outputs, lengths, mask = adapter(features, torch.tensor([2, 4]))
    assert torch.equal(outputs, torch.ones_like(features))
    assert lengths.tolist() == [2, 4]
    assert mask.dtype == torch.uint8


def test_compile_backend_uses_dynamic_fullgraph(monkeypatch):
    captured = {}

    def fake_compile(module, **kwargs):
        captured.update(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    CompileEncoderBackend(FakeEncoder(), profile="latency")
    assert captured == {
        "mode": "reduce-overhead",
        "dynamic": True,
        "fullgraph": True,
    }
```

- [ ] **Step 2: Run tests and confirm imports fail**

Run: `python3 -m pytest tests/fireredlid/test_encoder_backend.py -v`

Expected: collection fails because `encoder_backend.py` and `pytorch_backend.py` do not exist.

- [ ] **Step 3: Implement the backend contract and compatibility adapter**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EncoderResult:
    outputs: Tensor
    lengths: Tensor
    mask: Tensor


class EncoderBackend(torch.nn.Module, ABC):
    @abstractmethod
    def encode(self, features: Tensor, feature_lengths: Tensor) -> EncoderResult:
        raise NotImplementedError


class CompatibleEncoderAdapter(torch.nn.Module):
    def __init__(self, backend: EncoderBackend):
        super().__init__()
        self.backend = backend

    def forward(self, features: Tensor, feature_lengths: Tensor):
        result = self.backend.encode(features, feature_lengths)
        return result.outputs, result.lengths, result.mask
```

- [ ] **Step 4: Implement eager and compile backends**

```python
class EagerEncoderBackend(EncoderBackend):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def encode(self, features, feature_lengths):
        outputs, lengths, mask = self.encoder(features, feature_lengths)
        return EncoderResult(outputs=outputs, lengths=lengths, mask=mask)


class CompileEncoderBackend(EagerEncoderBackend):
    def __init__(self, encoder: torch.nn.Module, profile: str):
        mode = "reduce-overhead" if profile == "latency" else "max-autotune"
        compiled = torch.compile(
            encoder,
            mode=mode,
            dynamic=True,
            fullgraph=True,
        )
        super().__init__(compiled)
```

- [ ] **Step 5: Run backend tests**

Run: `python3 -m pytest tests/fireredlid/test_encoder_backend.py -v`

Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add fireredasr2s/fireredlid/runtime tests/fireredlid/test_encoder_backend.py
git commit -m "feat(fireredlid): add pluggable encoder backends"
```

---

### Task 5: Integrate config, backend selection, and physical batches

**Files:**
- Modify: `fireredasr2s/fireredlid/lid.py:15-110`
- Test: `tests/fireredlid/test_lid_runtime.py`

**Interfaces:**
- Consumes: `FeatExtractor.extract_many`, `BatchPlanner`, `CompatibleEncoderAdapter`, and PyTorch backends.
- Produces: extended `FireRedLidConfig`, backward-compatible eager behavior, and ordered results from multiple physical batches.

- [ ] **Step 1: Write config and reordered-result tests**

```python
from types import SimpleNamespace

import pytest
import torch

from fireredasr2s.fireredlid.lid import FireRedLid, FireRedLidConfig
from fireredasr2s.fireredlid.runtime.batch_planner import FeatureItem


class FakeTokenizer:
    def detokenize(self, ids):
        return str(ids[0])


class FakeModel:
    def process(self, features, lengths, *args):
        return [[{
            "yseq": torch.tensor([int(features[index, 0, 0].item())]),
            "confidence": torch.tensor(0.9),
        }] for index in range(features.size(0))]


def item(index, value, duration):
    return FeatureItem(
        index=index,
        uttid=f"utt-{index}",
        wav_input=f"{index}.wav",
        feature=torch.full((int(duration * 10), 80), float(value)),
        duration_s=duration,
        processed_duration_s=duration,
        truncated=False,
    )


def test_config_resolves_profile_defaults():
    assert FireRedLidConfig(profile="latency").resolved_batch_strategy == "none"
    assert FireRedLidConfig(profile="throughput").resolved_batch_strategy == "auto"


def test_tensorrt_requires_gpu_half_and_engine_dir():
    try:
        FireRedLidConfig(backend="tensorrt")
    except ValueError as error:
        assert "use_half=True" in str(error)
    else:
        raise AssertionError("TensorRT config must be rejected")


def test_only_eager_is_accepted_as_explicit_fallback():
    try:
        FireRedLidConfig(backend="compile", fallback_backend="compile")
    except ValueError as error:
        assert "fallback_backend" in str(error)
    else:
        raise AssertionError("non-eager fallback must be rejected")


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
    with pytest.raises(RuntimeError, match="failed to initialize compile backend"):
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
    lid = FireRedLid.__new__(FireRedLid)
    lid.model = FakeModel()
    lid.tokenizer = FakeTokenizer()
    lid.config = FireRedLidConfig(
        use_gpu=False,
        backend="eager",
        profile="throughput",
        batch_strategy="bucket",
        max_sub_batch_size=2,
    )
    results = lid._infer_items([item(0, 30, 30), item(1, 4, 4), item(2, 10, 10)])
    assert [result["uttid"] for result in results] == ["utt-0", "utt-1", "utt-2"]
    assert [result["lang"] for result in results] == ["30", "4", "10"]
```

- [ ] **Step 2: Run tests and verify config fields/helpers are missing**

Run: `python3 -m pytest tests/fireredlid/test_lid_runtime.py -v`

Expected: FAIL because the new config fields and `_infer_items()` do not exist.

- [ ] **Step 3: Extend and validate `FireRedLidConfig`**

```python
@dataclass
class FireRedLidConfig:
    use_gpu: bool = True
    use_half: bool = False
    backend: str = "eager"
    profile: str = "latency"
    max_audio_seconds: float = 60.0
    batch_strategy: str | None = None
    engine_dir: str | None = None
    max_sub_batch_size: int | None = None
    fallback_backend: str | None = None
    return_diagnostics: bool = False
    beam_size: int = field(init=False, default=3)
    nbest: int = field(init=False, default=1)
    decode_max_len: int = field(init=False, default=2)
    softmax_smoothing: float = field(init=False, default=1.25)
    aed_length_penalty: float = field(init=False, default=0.6)
    eos_penalty: float = field(init=False, default=1.0)

    def __post_init__(self):
        if self.backend not in {"eager", "compile", "tensorrt"}:
            raise ValueError(f"unsupported backend: {self.backend}")
        if self.profile not in {"latency", "throughput"}:
            raise ValueError(f"unsupported profile: {self.profile}")
        if self.max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be positive")
        if self.fallback_backend not in {None, "eager"}:
            raise ValueError("fallback_backend must be None or 'eager'")
        if self.backend == "tensorrt":
            if not self.use_gpu or not self.use_half or not self.engine_dir:
                raise ValueError("tensorrt requires use_gpu=True, use_half=True, and engine_dir")

    @property
    def resolved_batch_strategy(self):
        if self.batch_strategy is not None:
            return self.batch_strategy
        return "none" if self.profile == "latency" else "auto"
```

- [ ] **Step 4: Add backend selection before moving the model to its device**

Change `from_pretrained()` to pass `model_path` into `FireRedLid.__init__`, store it as `self.model_path`, and use it for TensorRT artifact compatibility checks. Keep the new constructor argument optional so direct test construction and existing internal callers remain compatible.

```python
def _configure_encoder_backend(self):
    self.backend_max_batch = None
    self.active_backend = "eager"
    if self.config.backend == "eager":
        return
    try:
        if self.config.backend == "compile":
            backend = CompileEncoderBackend(self.model.encoder, self.config.profile)
        else:
            from .runtime.tensorrt_backend import TensorRTEncoderBackend
            backend = TensorRTEncoderBackend(
                self.config.engine_dir,
                checkpoint_path=self.model_path,
            )
            self.backend_max_batch = backend.max_batch
    except Exception as error:
        if self.config.fallback_backend != "eager":
            raise RuntimeError(
                f"failed to initialize {self.config.backend} backend"
            ) from error
        logger.warning(
            "failed to initialize %s backend; using eager because fallback_backend='eager': %s",
            self.config.backend,
            error,
        )
        return
    self.model.encoder = CompatibleEncoderAdapter(backend)
    self.active_backend = self.config.backend
```

Call `_configure_encoder_backend()` before `model.half()` and `model.cuda()`. When TensorRT initializes successfully, remove the unused original PyTorch Encoder reference before device transfer; when explicit eager fallback is taken, preserve the original Encoder and move it with the rest of the model. Add a test that forces backend construction to raise and checks: without `fallback_backend` the wrapped initialization error is raised, while `fallback_backend="eager"` leaves `active_backend == "eager"`.

- [ ] **Step 5: Implement physical-batch inference and order restoration**

```python
def _infer_items(self, items):
    limits = [
        value for value in (
            self.config.max_sub_batch_size,
            self.backend_max_batch,
        ) if value is not None
    ]
    max_batch = min(limits) if limits else None
    planner = BatchPlanner(
        strategy=self.config.resolved_batch_strategy,
        max_sub_batch_size=max_batch,
    )
    raw_results = {}
    inference_elapsed = 0.0
    for planned in planner.plan(items):
        features = planned.padded_features
        lengths = planned.feature_lengths
        if self.config.use_gpu:
            features = features.cuda()
            lengths = lengths.cuda()
            if self.config.use_half:
                features = features.half()
        start_time = time.time()
        hypotheses = self.model.process(
            features,
            lengths,
            self.config.beam_size,
            self.config.nbest,
            self.config.decode_max_len,
            self.config.softmax_smoothing,
            self.config.aed_length_penalty,
            self.config.eos_penalty,
        )
        inference_elapsed += time.time() - start_time
        for item, hypotheses_for_item in zip(planned.items, hypotheses):
            hypothesis = hypotheses_for_item[0]
            ids = [int(token_id) for token_id in hypothesis["yseq"].cpu()]
            result = {
                "uttid": item.uttid,
                "lang": self.tokenizer.detokenize(ids),
                "confidence": round(hypothesis["confidence"].cpu().item(), 3),
                "dur_s": round(item.duration_s, 3),
            }
            if isinstance(item.wav_input, str):
                result["wav"] = item.wav_input
            if self.config.return_diagnostics:
                result.update({
                    "backend": self.active_backend,
                    "truncated": item.truncated,
                    "processed_dur_s": round(item.processed_duration_s, 3),
                })
            raw_results[item.index] = result
    total_duration = sum(item.duration_s for item in items)
    rtf = inference_elapsed / total_duration if total_duration else 0.0
    ordered = []
    for item in sorted(items, key=lambda value: value.index):
        result = raw_results[item.index]
        result["rtf"] = f"{rtf:.4f}"
        ordered.append(result)
    return ordered
```

Update `process()` to call `extract_many(..., max_audio_seconds=config.max_audio_seconds)` and then `_infer_items()`. Preserve the existing feature-extraction exception handling and the empty-feature return.

- [ ] **Step 6: Run runtime, feature, planner, and backend tests**

Run: `python3 -m pytest tests/fireredlid/test_lid_runtime.py tests/fireredlid/test_feat.py tests/fireredlid/test_batch_planner.py tests/fireredlid/test_encoder_backend.py -v`

Expected: `13 passed`.

- [ ] **Step 7: Commit**

```bash
git add fireredasr2s/fireredlid/lid.py tests/fireredlid/test_lid_runtime.py
git commit -m "feat(fireredlid): add configurable inference backends"
```

---

### Task 6: Export and verify a dynamic FP32 ONNX Encoder on macOS

**Files:**
- Modify: `.gitignore`
- Create: `runtime/fireredlid/pyproject.toml`
- Create: `runtime/fireredlid/export_encoder_onnx.py`
- Create: `runtime/fireredlid/verify.py`
- Test: `tests/fireredlid/test_onnx_export.py`

**Interfaces:**
- Produces: `EncoderExportWrapper`, `export_encoder(...)`, and `verify_onnx_outputs(...)`.
- Consumers: Task 7 TensorRT builder and Task 8 documented commands.

- [ ] **Step 1: Define the isolated Mac export environment**

```toml
[project]
name = "fireredlid-runtime-tools"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "onnx==1.19.0",
    "onnxruntime==1.21.0",
    "PyYAML==6.0.2",
]

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 2: Write a dynamic ONNX export test with a tiny Encoder**

```python
import importlib.util
from pathlib import Path

import onnxruntime as ort
import torch


SCRIPT = Path("runtime/fireredlid/export_encoder_onnx.py")


class TinyEncoder(torch.nn.Module):
    def forward(self, features, lengths):
        mask = (
            torch.arange(features.size(1))[None, :] < lengths[:, None]
        ).unsqueeze(1).to(torch.uint8)
        return features * 2.0, lengths, mask


def load_export_module():
    spec = importlib.util.spec_from_file_location("fireredlid_export", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_supports_dynamic_batch_and_time(tmp_path):
    module = load_export_module()
    path = tmp_path / "encoder.onnx"
    module.export_encoder(TinyEncoder().eval(), path, torch.zeros(1, 4, 80), torch.tensor([4]))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {
        "features": torch.zeros(2, 7, 80).numpy(),
        "feature_lengths": torch.tensor([5, 7]).numpy(),
    })
    assert outputs[0].shape == (2, 7, 80)
    assert outputs[1].shape == (2,)
    assert outputs[2].shape == (2, 1, 7)
```

- [ ] **Step 3: Run the test and confirm the script is missing**

Run: `python3 -m pytest tests/fireredlid/test_onnx_export.py -v`

Expected: FAIL with `FileNotFoundError` for `export_encoder_onnx.py`.

- [ ] **Step 4: Implement the export wrapper and dynamic axes**

```python
class EncoderExportWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, features, feature_lengths):
        return self.encoder(features, feature_lengths)


def export_encoder(encoder, output_path, sample_features, sample_lengths):
    wrapper = EncoderExportWrapper(encoder.float().cpu().eval())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (sample_features.float().cpu(), sample_lengths.long().cpu()),
        str(output_path),
        input_names=["features", "feature_lengths"],
        output_names=["encoder_outputs", "encoder_lengths", "encoder_mask"],
        dynamic_axes={
            "features": {0: "batch", 1: "frames"},
            "feature_lengths": {0: "batch"},
            "encoder_outputs": {0: "batch", 1: "encoder_frames"},
            "encoder_lengths": {0: "batch"},
            "encoder_mask": {0: "batch", 2: "encoder_frames"},
        },
        opset_version=17,
        do_constant_folding=True,
        external_data=True,
    )
```

The CLI loads `FireRedLID/model.pth.tar`, uses sample shape `[1, 1000, 80]`, and writes `encoder.fp32.onnx`. Keep external tensor data next to the ONNX file when the model exceeds protobuf size limits.

- [ ] **Step 5: Implement ONNX Runtime verification**

```python
def verify_onnx_outputs(encoder, session, features, lengths):
    with torch.inference_mode():
        expected_outputs, expected_lengths, expected_mask = encoder(features, lengths)
    actual_outputs, actual_lengths, actual_mask = session.run(None, {
        "features": features.cpu().numpy(),
        "feature_lengths": lengths.cpu().numpy(),
    })
    actual_outputs = torch.from_numpy(actual_outputs)
    actual_lengths = torch.from_numpy(actual_lengths)
    actual_mask = torch.from_numpy(actual_mask)
    if not torch.equal(actual_mask, expected_mask.cpu()):
        raise AssertionError("encoder_mask differs from eager baseline")
    if not torch.equal(actual_lengths, expected_lengths.cpu()):
        raise AssertionError("encoder_lengths differ from eager baseline")
    torch.testing.assert_close(actual_outputs, expected_outputs.cpu(), rtol=1e-3, atol=1e-4)
```

The CLI verifies frame lengths corresponding to 1, 5, 15, 30, and 60 seconds with batch sizes `1` and `2`; it writes a JSON report and exits nonzero on any mismatch.

- [ ] **Step 6: Run the tiny export test and script help**

Run: `python3 -m pytest tests/fireredlid/test_onnx_export.py -v`

Expected: `2 passed`.

Run: `python3 runtime/fireredlid/export_encoder_onnx.py --help`

Expected: exit code `0` and options for `--model-dir` and `--output-dir`.

- [ ] **Step 7: Commit**

```bash
git add .gitignore runtime/fireredlid/pyproject.toml runtime/fireredlid/export_encoder_onnx.py runtime/fireredlid/verify.py tests/fireredlid/test_onnx_export.py
git commit -m "feat(fireredlid): add dynamic ONNX encoder export"
```

---

### Task 7: Add TensorRT profiles, manifest validation, and Linux builder surface

**Files:**
- Create: `runtime/fireredlid/profiles.yaml`
- Create: `runtime/fireredlid/build_engine.py`
- Create: `fireredasr2s/fireredlid/runtime/tensorrt_backend.py`
- Modify: `fireredasr2s/fireredlid/runtime/__init__.py`
- Test: `tests/fireredlid/test_tensorrt_backend.py`

**Interfaces:**
- Produces: `EngineManifest.load()`, `TensorRTEncoderBackend.max_batch`, lazy platform failure, shape-range validation, and `build_engine(...)`.
- Consumes: ONNX input/output names from Task 6 and `EncoderResult` from Task 4.

- [ ] **Step 1: Create the initial profile**

```yaml
schema_version: 1
profiles:
  - name: default
    features:
      min: [1, 1, 80]
      opt: [1, 1000, 80]
      max: [4, 6000, 80]
    feature_lengths:
      min: [1]
      opt: [1]
      max: [4]
```

- [ ] **Step 2: Write manifest and lazy-import tests**

```python
import hashlib
import json

import pytest

from fireredasr2s.fireredlid.runtime.tensorrt_backend import (
    ArtifactMismatchError,
    BackendUnavailableError,
    EngineManifest,
    TensorRTEncoderBackend,
)


def test_manifest_exposes_max_batch(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "precision": "float16",
        "checkpoint_sha256": "0" * 64,
        "onnx_sha256": "1" * 64,
        "input_names": ["features", "feature_lengths"],
        "output_names": ["encoder_outputs", "encoder_lengths", "encoder_mask"],
        "profiles": [{"name": "default", "min": [1, 1, 80], "opt": [1, 1000, 80], "max": [4, 6000, 80]}],
    }))
    manifest = EngineManifest.load(manifest_path)
    assert manifest.max_batch == 4


def test_tensorrt_backend_fails_cleanly_without_tensorrt(tmp_path):
    (tmp_path / "encoder.plan").write_bytes(b"not-an-engine")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "precision": "float16",
        "checkpoint_sha256": "0" * 64,
        "onnx_sha256": "1" * 64,
        "input_names": ["features", "feature_lengths"],
        "output_names": ["encoder_outputs", "encoder_lengths", "encoder_mask"],
        "profiles": [{"name": "default", "min": [1, 1, 80], "opt": [1, 1000, 80], "max": [4, 6000, 80]}],
    }))
    with pytest.raises(BackendUnavailableError, match="Linux NVIDIA"):
        TensorRTEncoderBackend(tmp_path)


def test_manifest_rejects_checkpoint_mismatch(tmp_path):
    checkpoint = tmp_path / "model.pth.tar"
    checkpoint.write_bytes(b"checkpoint-a")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "precision": "float16",
        "checkpoint_sha256": hashlib.sha256(b"checkpoint-b").hexdigest(),
        "onnx_sha256": "1" * 64,
        "input_names": ["features", "feature_lengths"],
        "output_names": ["encoder_outputs", "encoder_lengths", "encoder_mask"],
        "profiles": [{"name": "default", "min": [1, 1, 80], "opt": [1, 1000, 80], "max": [4, 6000, 80]}],
    }))
    manifest = EngineManifest.load(manifest_path)
    with pytest.raises(ArtifactMismatchError, match="checkpoint SHA-256"):
        manifest.validate_checkpoint(checkpoint)
```

- [ ] **Step 3: Run tests and confirm the backend module is missing**

Run: `python3 -m pytest tests/fireredlid/test_tensorrt_backend.py -v`

Expected: collection fails because `tensorrt_backend.py` does not exist.

- [ ] **Step 4: Implement manifest loading and lazy platform checks**

```python
@dataclass(frozen=True)
class EngineManifest:
    precision: str
    checkpoint_sha256: str
    onnx_sha256: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    profiles: tuple[dict, ...]

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") != 1:
            raise ArtifactMismatchError("unsupported engine manifest schema")
        expected_outputs = ["encoder_outputs", "encoder_lengths", "encoder_mask"]
        if data.get("output_names") != expected_outputs:
            raise ArtifactMismatchError("engine output contract mismatch")
        return cls(
            precision=data["precision"],
            checkpoint_sha256=data["checkpoint_sha256"],
            onnx_sha256=data["onnx_sha256"],
            input_names=tuple(data["input_names"]),
            output_names=tuple(data["output_names"]),
            profiles=tuple(data["profiles"]),
        )

    @property
    def max_batch(self):
        return max(profile["max"][0] for profile in self.profiles)

    def validate_checkpoint(self, checkpoint_path):
        actual = sha256_file(checkpoint_path)
        if actual != self.checkpoint_sha256:
            raise ArtifactMismatchError("checkpoint SHA-256 differs from engine manifest")
```

Implement `sha256_file()` as a streaming 1 MiB chunk reader. In `TensorRTEncoderBackend.__init__(engine_dir, checkpoint_path=None)`, load and validate the manifest first, verify the checkpoint hash when a path is supplied, then raise `BackendUnavailableError("TensorRT backend requires Linux NVIDIA with TensorRT installed")` when `sys.platform != "linux"`, `torch.cuda.is_available()` is false, or `importlib.util.find_spec("tensorrt")` is absent. Do not import TensorRT at module import time.

- [ ] **Step 5: Implement the Linux execution surface without claiming Mac execution**

```python
def encode(self, features, feature_lengths):
    features = features.to(device="cuda", dtype=torch.float16).contiguous()
    feature_lengths = feature_lengths.to(device="cuda", dtype=torch.int32).contiguous()
    self._validate_shape(tuple(features.shape))
    self.context.set_input_shape("features", tuple(features.shape))
    self.context.set_input_shape("feature_lengths", tuple(feature_lengths.shape))
    output_shape = tuple(self.context.get_tensor_shape("encoder_outputs"))
    lengths_shape = tuple(self.context.get_tensor_shape("encoder_lengths"))
    mask_shape = tuple(self.context.get_tensor_shape("encoder_mask"))
    outputs = torch.empty(output_shape, device="cuda", dtype=torch.float16)
    lengths = torch.empty(lengths_shape, device="cuda", dtype=torch.int64)
    mask = torch.empty(mask_shape, device="cuda", dtype=torch.uint8)
    tensors = {
        "features": features,
        "feature_lengths": feature_lengths,
        "encoder_outputs": outputs,
        "encoder_lengths": lengths,
        "encoder_mask": mask,
    }
    for name, tensor in tensors.items():
        self.context.set_tensor_address(name, tensor.data_ptr())
    stream = torch.cuda.current_stream()
    if not self.context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT encoder execution failed")
    return EncoderResult(outputs=outputs, lengths=lengths, mask=mask)
```

Engine deserialization follows the existing ASR runtime pattern: read `encoder.plan`, construct `trt.Runtime`, deserialize, create one execution context, and validate all five tensor names against the manifest. Before allocating outputs, validate TensorRT reports `encoder_lengths` as `INT64` and `encoder_mask` as `UINT8`; raise `ArtifactMismatchError` instead of silently casting an incompatible engine.

- [ ] **Step 6: Implement the Linux-only builder**

`build_engine.py` must:

```python
def build_engine(onnx_path, output_dir, profile_path, checkpoint_path):
    trt = require_tensorrt()
    profile_config = load_profile(profile_path)
    builder = trt.Builder(trt.Logger(trt.Logger.INFO))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt.Logger(trt.Logger.INFO))
    if not parser.parse(Path(onnx_path).read_bytes()):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(errors)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    for item in profile_config["profiles"]:
        profile = builder.create_optimization_profile()
        profile.set_shape("features", tuple(item["features"]["min"]), tuple(item["features"]["opt"]), tuple(item["features"]["max"]))
        profile.set_shape("feature_lengths", tuple(item["feature_lengths"]["min"]), tuple(item["feature_lengths"]["opt"]), tuple(item["feature_lengths"]["max"]))
        config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT returned an empty serialized engine")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "encoder.plan").write_bytes(serialized)
    write_manifest(output_dir, profile_config, checkpoint_path, onnx_path, trt.__version__)
```

`write_manifest()` stores SHA-256 for the checkpoint and ONNX, GPU name, PyTorch/TensorRT/CUDA versions, precision, tensor names, and flattened min/opt/max profile shapes.

- [ ] **Step 7: Run Mac-safe tests and CLI help**

Run: `python3 -m pytest tests/fireredlid/test_tensorrt_backend.py -v`

Expected: `3 passed` without importing TensorRT.

Run: `python3 runtime/fireredlid/build_engine.py --help`

Expected: exit code `0`; executing without Linux TensorRT must produce the explicit platform capability error.

- [ ] **Step 8: Commit**

```bash
git add runtime/fireredlid/profiles.yaml runtime/fireredlid/build_engine.py fireredasr2s/fireredlid/runtime tests/fireredlid/test_tensorrt_backend.py
git commit -m "feat(fireredlid): add TensorRT engine contract"
```

---

### Task 8: Add verification, benchmark reporting, and handoff documentation

**Files:**
- Modify: `fireredasr2s/fireredlid/lid.py`
- Modify: `fireredasr2s/fireredlid/models/fireredlid_aed.py`
- Create: `runtime/fireredlid/benchmark.py`
- Create: `runtime/fireredlid/example_manifest.jsonl`
- Create: `runtime/fireredlid/README.md`
- Modify: `runtime/fireredlid/verify.py`
- Test: `tests/fireredlid/test_benchmark.py`

**Interfaces:**
- Produces: JSON reports with P50/P95/P99, utterances/s, audio-seconds/s, RTF, physical-batch count, and padding ratio.
- Documents: exact Mac verification commands and Linux GPU handoff commands.

- [ ] **Step 1: Write deterministic metric tests**

```python
import importlib.util
from pathlib import Path


SCRIPT = Path("runtime/fireredlid/benchmark.py")


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("fireredlid_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_reports_percentiles_and_throughput():
    module = load_benchmark_module()
    summary = module.summarize(
        latencies_s=[0.01, 0.02, 0.03, 0.04, 0.05],
        utterances=10,
        audio_seconds=100.0,
        elapsed_s=2.0,
    )
    assert summary["p50_ms"] == 30.0
    assert summary["p95_ms"] == 48.0
    assert summary["utterances_per_s"] == 5.0
    assert summary["audio_seconds_per_s"] == 50.0
    assert summary["rtf"] == 0.02


def test_stage_recorder_reports_named_stages():
    module = load_benchmark_module()
    recorder = module.StageRecorder(synchronize_cuda=False)
    with recorder.measure("encoder"):
        pass
    assert set(recorder.total_seconds) == {"encoder"}
    assert recorder.total_seconds["encoder"] >= 0.0
```

- [ ] **Step 2: Run the test and confirm the script is missing**

Run: `python3 -m pytest tests/fireredlid/test_benchmark.py -v`

Expected: FAIL with `FileNotFoundError` for `benchmark.py`.

- [ ] **Step 3: Implement stable summary calculations and JSON output**

```python
def summarize(latencies_s, utterances, audio_seconds, elapsed_s):
    values_ms = np.asarray(latencies_s, dtype=np.float64) * 1000.0
    return {
        "p50_ms": round(float(np.percentile(values_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(values_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(values_ms, 99)), 3),
        "utterances_per_s": round(utterances / elapsed_s, 3),
        "audio_seconds_per_s": round(audio_seconds / elapsed_s, 3),
        "rtf": round(elapsed_s / audio_seconds, 6),
    }
```

Add a `StageRecorder` context manager using `time.perf_counter()`. Its `measure(name)` method synchronizes CUDA immediately before and after a measured region when `synchronize_cuda=True`, then accumulates seconds by stage name. Keep it independent of FireRedLID imports so the metrics unit test remains lightweight.

Add opt-in internal timing hooks without changing the public `process()` return value:

- `FireRedLid.process()` measures `fbank` around `extract_many()`.
- `FireRedLid._infer_items()` measures `h2d` around device/dtype transfer and `result_formatting` around token/result conversion.
- `FireRedLidAed.process(..., stage_recorder=None)` measures `encoder` around `self.encoder(...)` and `decoder` around `batch_beam_search(...)`.
- When no recorder is installed, use `contextlib.nullcontext()` and preserve the original call path.

The benchmark CLI installs the recorder on the loaded `FireRedLid` instance, accepts `--backend`, `--profile`, `--batch-strategy`, `--manifest`, `--warmup`, `--iterations`, and `--scope encoder|model|end-to-end`. It excludes warm-up and first compile from stable-state metrics; when CUDA is active it calls `torch.cuda.reset_peak_memory_stats()` immediately before stable iterations and reports `torch.cuda.max_memory_allocated()`. It writes arguments, environment versions, logical/physical batch counts, padding ratio, actual input shapes, stage totals, first-run time, warm-up time, stable-state metrics, and peak GPU bytes (`null` on CPU) into the JSON report.

Run: `python3 -m pytest tests/fireredlid/test_benchmark.py -v`

Expected: `2 passed`.

- [ ] **Step 4: Document exact Mac and Linux commands**

The README must contain these Mac commands:

```bash
python3 -m pip install "git+https://github.com/csukuangfj/kaldi-native-fbank.git@f68c6b43f739697d7ab02ff6debacee130e1d541"
python3 -m pytest tests/fireredlid -v
python3 runtime/fireredlid/export_encoder_onnx.py --model-dir FireRedLID --output-dir runtime/fireredlid/artifacts
python3 runtime/fireredlid/verify.py --model-dir FireRedLID --onnx runtime/fireredlid/artifacts/encoder.fp32.onnx
python3 runtime/fireredlid/benchmark.py --model-dir FireRedLID --backend eager --profile latency --scope end-to-end --manifest runtime/fireredlid/example_manifest.jsonl
```

It must state that `runtime/fireredlid/artifacts/` is untracked and contain these Linux handoff commands:

```bash
python3 runtime/fireredlid/build_engine.py --onnx runtime/fireredlid/artifacts/encoder.fp32.onnx --checkpoint FireRedLID/model.pth.tar --profiles runtime/fireredlid/profiles.yaml --output-dir runtime/fireredlid/artifacts/engine
python3 runtime/fireredlid/verify.py --model-dir FireRedLID --engine-dir runtime/fireredlid/artifacts/engine --backend tensorrt
python3 runtime/fireredlid/benchmark.py --model-dir FireRedLID --backend tensorrt --profile latency --scope end-to-end --manifest runtime/fireredlid/example_manifest.jsonl
python3 runtime/fireredlid/benchmark.py --model-dir FireRedLID --backend tensorrt --profile throughput --batch-strategy auto --scope end-to-end --manifest runtime/fireredlid/example_manifest.jsonl
```

Explicitly label TensorRT, CUDA `torch.compile`, and RTX PRO 5000/L20 performance as unverified on Mac.

Create `runtime/fireredlid/example_manifest.jsonl` with exactly:

```jsonl
{"uttid":"hello_zh","wav":"assets/hello_zh.wav"}
{"uttid":"hello_en","wav":"assets/hello_en.wav"}
```

Confirm `/runtime/fireredlid/artifacts/` remains in `.gitignore`; do not ignore `example_manifest.jsonl`, `profiles.yaml`, or the runtime source files.

- [ ] **Step 5: Run the full Mac-safe unit suite**

Run: `python3 -m pytest tests/fireredlid -v`

Expected: all tests pass; TensorRT tests exercise only lazy failure and manifest validation.

Run: `python3 -m compileall -q fireredasr2s/fireredlid runtime/fireredlid`

Expected: exit code `0`.

- [ ] **Step 6: Run the real-model eager smoke test**

Run:

```bash
python3 -c 'from fireredasr2s.fireredlid.lid import FireRedLid, FireRedLidConfig; lid=FireRedLid.from_pretrained("FireRedLID", FireRedLidConfig(use_gpu=False)); print(lid.process(["zh", "en"], ["assets/hello_zh.wav", "assets/hello_en.wav"]))'
```

Expected: two results in input order; the first language is `zh mandarin`, the second is `en`; confidence values are finite and between `0` and `1`.

- [ ] **Step 7: Run real-model ONNX export and FP32 verification**

Run:

```bash
python3 runtime/fireredlid/export_encoder_onnx.py --model-dir FireRedLID --output-dir runtime/fireredlid/artifacts
python3 runtime/fireredlid/verify.py --model-dir FireRedLID --onnx runtime/fireredlid/artifacts/encoder.fp32.onnx
```

Expected: export completes with dynamic inputs; lengths and mask are independently exact against eager; Encoder outputs pass `rtol=1e-3, atol=1e-4`. If the full 60-second case exceeds available Mac memory, the verifier reports that case as a resource limitation and still exits nonzero; do not mark Mac Phase A complete until the case runs on a machine with sufficient RAM.

- [ ] **Step 8: Commit**

```bash
git add fireredasr2s/fireredlid/lid.py fireredasr2s/fireredlid/models/fireredlid_aed.py runtime/fireredlid/benchmark.py runtime/fireredlid/verify.py runtime/fireredlid/example_manifest.jsonl runtime/fireredlid/README.md tests/fireredlid/test_benchmark.py
git commit -m "test(fireredlid): add inference verification and benchmarks"
```

---

## Final Mac Phase Verification

Run all of the following after Task 8:

```bash
git diff --check
python3 -m pytest tests/fireredlid -v
python3 -m compileall -q fireredasr2s/fireredlid runtime/fireredlid
git status --short
```

Expected:

- `git diff --check` has no output.
- Every FireRedLID test passes.
- Compileall exits `0`.
- `FireRedLID/` and generated `runtime/fireredlid/artifacts/` remain untracked/ignored and are never staged.
- No result claims TensorRT, CUDA `torch.compile`, RTX PRO 5000, or L20 as verified.
