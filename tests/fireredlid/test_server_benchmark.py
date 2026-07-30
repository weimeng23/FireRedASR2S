import base64
import importlib.util
import json
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path("runtime/fireredlid/server_benchmark.py")


def load_server_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "fireredlid_server_benchmark",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_manifest_prepares_real_wav_once(tmp_path):
    module = load_server_benchmark_module()
    audio_path = tmp_path / "sample.wav"
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 1600)
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        json.dumps({"uttid": "sample", "wav": str(audio_path)}) + "\n",
        encoding="utf-8",
    )

    records = module.load_manifest(manifest)

    assert records == [
        {
            "uttid": "sample",
            "wav": str(audio_path),
            "audio_base64": base64.b64encode(
                audio_path.read_bytes()
            ).decode("ascii"),
            "duration_s": 0.1,
        }
    ]


def test_run_benchmark_enforces_concurrency_and_reports_success_metrics():
    module = load_server_benchmark_module()
    records = [
        {
            "uttid": "one",
            "wav": "one.wav",
            "audio_base64": "one",
            "duration_s": 1.0,
        },
        {
            "uttid": "two",
            "wav": "two.wav",
            "audio_base64": "two",
            "duration_s": 2.0,
        },
    ]
    active = 0
    max_active = 0
    calls = 0
    lock = threading.Lock()

    def send_request(url, payload, timeout_s):
        nonlocal active, max_active, calls
        assert url == "http://server/v1/lid"
        assert timeout_s == 30.0
        request = json.loads(payload)
        with lock:
            calls += 1
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return (
            {
                "backend": "eager",
                "dtype": "mixed",
                "encoder_dtype": "float16",
                "decoder_dtype": "float32",
                "results": [
                    {
                        "uttid": item["uttid"],
                        "lang": "en",
                        "confidence": 0.9,
                    }
                    for item in request["inputs"]
                ],
            },
            0.02,
        )

    times = iter((10.0, 12.0))
    report = module.run_benchmark(
        records,
        url="http://server/v1/lid",
        concurrency=3,
        total_requests=6,
        request_batch_size=2,
        warmup_requests=2,
        timeout_s=30.0,
        request_sender=send_request,
        clock=lambda: next(times),
    )

    assert calls == 8
    assert max_active == 3
    assert report["server"] == {
        "backend": "eager",
        "dtype": "mixed",
        "encoder_dtype": "float16",
        "decoder_dtype": "float32",
    }
    assert report["summary"] == {
        "elapsed_s": 2.0,
        "requests_total": 6,
        "requests_succeeded": 6,
        "requests_failed": 0,
        "request_error_rate": 0.0,
        "requests_per_s": 3.0,
        "items_total": 12,
        "items_succeeded": 12,
        "items_failed": 0,
        "items_per_s": 6.0,
        "audio_seconds_succeeded": 18.0,
        "audio_seconds_per_s": 9.0,
        "request_latency_ms": {
            "min": 20.0,
            "mean": 20.0,
            "p50": 20.0,
            "p95": 20.0,
            "p99": 20.0,
            "max": 20.0,
        },
    }
    assert report["errors"] == []


def test_run_benchmark_records_failed_requests_without_stopping():
    module = load_server_benchmark_module()
    records = [
        {
            "uttid": "one",
            "wav": "one.wav",
            "audio_base64": "one",
            "duration_s": 1.0,
        }
    ]

    def send_request(unused_url, payload, unused_timeout):
        request = json.loads(payload)
        uttid = request["inputs"][0]["uttid"]
        if "-request-1-" in uttid:
            raise RuntimeError("connection reset")
        return (
            {
                "backend": "eager",
                "dtype": "mixed",
                "encoder_dtype": "float16",
                "decoder_dtype": "float32",
                "results": [{"uttid": uttid, "lang": "en"}],
            },
            0.01,
        )

    times = iter((5.0, 6.0))
    report = module.run_benchmark(
        records,
        url="http://server/v1/lid",
        concurrency=2,
        total_requests=3,
        request_batch_size=1,
        warmup_requests=0,
        timeout_s=30.0,
        request_sender=send_request,
        clock=lambda: next(times),
    )

    assert report["summary"]["requests_succeeded"] == 2
    assert report["summary"]["requests_failed"] == 1
    assert report["summary"]["request_error_rate"] == pytest.approx(
        1 / 3,
        abs=1e-6,
    )
    assert report["summary"]["items_succeeded"] == 2
    assert report["summary"]["items_failed"] == 1
    assert report["errors"] == [
        {
            "request_index": 1,
            "error_type": "RuntimeError",
            "message": "connection reset",
        }
    ]


def test_main_writes_json_report_and_returns_success(
    tmp_path,
    monkeypatch,
):
    module = load_server_benchmark_module()
    output = tmp_path / "reports" / "server.json"
    args = SimpleNamespace(
        manifest="input.jsonl",
        url="http://server/v1/lid",
        concurrency=8,
        requests=10,
        request_batch_size=1,
        warmup_requests=2,
        timeout=30.0,
        output=str(output),
    )
    records = [{"uttid": "one"}]
    report = {
        "schema_version": 1,
        "summary": {"requests_failed": 0},
        "errors": [],
    }
    monkeypatch.setattr(module, "parse_args", lambda unused_argv: args)
    monkeypatch.setattr(
        module,
        "load_manifest",
        lambda unused_path: records,
    )
    monkeypatch.setattr(
        module,
        "run_benchmark",
        lambda *unused_args, **unused_kwargs: report,
    )

    exit_code = module.main([])

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        **report,
        "manifest": {
            "path": "input.jsonl",
            "records": 1,
        },
    }


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--concurrency", "0"),
        ("--requests", "0"),
        ("--request-batch-size", "0"),
        ("--warmup-requests", "-1"),
        ("--timeout", "0"),
    ],
)
def test_parse_args_rejects_invalid_workload_values(option, value):
    module = load_server_benchmark_module()

    with pytest.raises(SystemExit) as raised:
        module.parse_args(
            [
                "--manifest",
                "input.jsonl",
                option,
                value,
            ]
        )

    assert raised.value.code == 2
