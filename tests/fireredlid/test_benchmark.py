import importlib.util
from pathlib import Path


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
