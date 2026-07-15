import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fireredasr2s.fireredlid import lid as lid_module


SCRIPT = Path("runtime/fireredlid/benchmark.py")


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "fireredlid_benchmark",
        SCRIPT,
    )
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
    assert summary["p99_ms"] == 49.6
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


def test_load_manifest_preserves_uttids_and_wave_paths(tmp_path):
    module = load_benchmark_module()
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        '{"uttid":"a","wav":"a.wav"}\n'
        '{"uttid":"b","wav":"b.wav"}\n',
        encoding="utf-8",
    )

    items = module.load_manifest(manifest)

    assert items == [
        {"uttid": "a", "wav": "a.wav"},
        {"uttid": "b", "wav": "b.wav"},
    ]


def test_benchmark_report_includes_common_provenance_and_real_input_hashes(
    tmp_path,
    monkeypatch,
):
    module = load_benchmark_module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_files = {
        "checkpoint": model_dir / "model.pth.tar",
        "cmvn": model_dir / "cmvn.ark",
        "dictionary": model_dir / "dict.txt",
    }
    for name, path in model_files.items():
        path.write_bytes(name.encode("utf-8"))
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        '{"uttid":"a","wav":"a.wav"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "benchmark.json"
    args = SimpleNamespace(
        model_dir=str(model_dir),
        manifest=str(manifest),
        backend="eager",
        profile="latency",
        scope="encoder",
        batch_strategy="none",
        device="cpu",
        precision="fp32",
        engine_dir=None,
        max_sub_batch_size=None,
        logical_batch_size=1,
        max_audio_seconds=60.0,
        warmup=0,
        iterations=1,
        output=str(output),
    )

    class FakeExtractor:
        def extract_many(self, *unused_args, **unused_kwargs):
            return [SimpleNamespace(duration_s=1.0)]

    fake_lid = SimpleNamespace(
        feat_extractor=FakeExtractor(),
        active_backend="eager",
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(
        lid_module.FireRedLid,
        "from_pretrained",
        staticmethod(lambda unused_dir, config: fake_lid),
    )
    monkeypatch.setattr(
        module,
        "_plan_report",
        lambda unused_lid, unused_groups: {
            "logical_batch_count": 1,
            "physical_batch_count": 1,
            "actual_input_shapes": [[1, 100, 80]],
            "valid_frames": 100,
            "padded_frames": 100,
            "padding_ratio": 0.0,
        },
    )
    monkeypatch.setattr(
        module,
        "_run_workload",
        lambda *unused_args, **unused_kwargs: ([0.01], 1, 1.0),
    )

    module.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    artifacts = report["provenance"]["input_artifacts"]
    assert set(artifacts) == {
        "checkpoint",
        "cmvn",
        "dictionary",
        "audio_manifest",
    }
    for name, path in model_files.items():
        assert artifacts[name]["sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    assert artifacts["audio_manifest"]["sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert report["provenance"]["arguments"]["device"] == "cpu"
    assert report["requested_device"] == "cpu"
    assert report["resolved_device"] == "cpu"
    assert report["requested_precision"] == "fp32"
    assert report["resolved_precision"] == "fp32"
    json.dumps(report["provenance"])


@pytest.mark.parametrize(
    ("device", "precision"),
    [
        ("auto", "fp16"),
        ("cpu", "fp16"),
        ("cuda", "fp32"),
        ("cpu", "fp32"),
    ],
)
def test_benchmark_cli_rejects_non_cuda_fp16_tensorrt_selection(
    monkeypatch,
    device,
    precision,
):
    module = load_benchmark_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark.py",
            "--model-dir",
            "model",
            "--manifest",
            "input.jsonl",
            "--backend",
            "tensorrt",
            "--engine-dir",
            "engine",
            "--device",
            device,
            "--precision",
            precision,
        ],
    )

    with pytest.raises(SystemExit) as error:
        module.parse_args()

    assert error.value.code == 2


def test_benchmark_rejects_invalid_tensorrt_config_before_any_side_effect(
    tmp_path,
    monkeypatch,
):
    module = load_benchmark_module()
    output = tmp_path / "reports" / "benchmark.json"
    args = SimpleNamespace(
        model_dir="missing-model",
        manifest="missing-manifest.jsonl",
        backend="tensorrt",
        profile="latency",
        scope="end-to-end",
        batch_strategy=None,
        device="cpu",
        precision="fp32",
        engine_dir="missing-engine",
        max_sub_batch_size=None,
        logical_batch_size=None,
        max_audio_seconds=60.0,
        warmup=1,
        iterations=1,
        output=str(output),
    )
    side_effects = []
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(
        module,
        "load_manifest",
        lambda unused_path: side_effects.append("manifest"),
    )
    monkeypatch.setattr(
        lid_module.FireRedLid,
        "from_pretrained",
        staticmethod(
            lambda unused_dir, unused_config: side_effects.append("model")
        ),
    )

    with pytest.raises(ValueError, match="device=cuda.*precision=fp16"):
        module.main()

    assert side_effects == []
    assert not output.parent.exists()
