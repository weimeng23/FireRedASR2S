import base64
import io
import inspect
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from fastapi.testclient import TestClient

import fireredasr2s.fireredlid.server as server_module
from fireredasr2s.fireredlid.server import (
    BucketPolicy,
    ServerSettings,
    create_app,
    create_lid_config,
    main,
    parse_settings,
)


class FakeLid:
    active_backend = "compile"

    def process(self, uttids, wav_inputs):
        return [
            {"uttid": uttid, "lang": "en", "confidence": 0.99}
            for uttid in uttids
        ]


def encode_wav(*, channels=1, sample_rate=16000, sample_count=1600):
    samples = np.arange(sample_count, dtype=np.int16)
    if channels == 2:
        samples = np.column_stack((samples, samples))
    buffer = io.BytesIO()
    sf.write(
        buffer,
        samples,
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_server_settings_use_dynamic_batching_and_mixed_precision_defaults():
    settings = ServerSettings(model_dir="/models/FireRedLID")

    assert settings.port == 12345
    assert settings.queue_capacity == 512
    assert settings.decode_workers == 8
    assert settings.max_batch_delay_ms == 5.0
    assert settings.encoder_precision == "fp16"
    assert settings.decoder_precision == "fp32"
    assert not hasattr(settings, "max_sub_batch_size")
    assert [
        (policy.max_seconds, policy.max_batch_size)
        for policy in settings.bucket_policies
    ] == [(5.0, 32), (15.0, 16), (30.0, 8), (60.0, 4)]


def test_server_settings_preserve_runtime_backend_configuration():
    settings = ServerSettings(
        model_dir="/models/FireRedLID",
        backend="tensorrt",
        profile="throughput",
        use_gpu=True,
        encoder_precision="fp16",
        decoder_precision="fp32",
        engine_dir="/engines/l20",
        fallback_backend="eager",
        max_audio_seconds=30.0,
        bucket_policies=(BucketPolicy(30, 4),),
    )

    config = create_lid_config(settings)

    assert config.backend == "tensorrt"
    assert config.profile == "throughput"
    assert config.use_gpu is True
    assert config.encoder_precision == "fp16"
    assert config.decoder_precision == "fp32"
    assert config.engine_dir == "/engines/l20"
    assert config.fallback_backend == "eager"
    assert config.max_audio_seconds == 30.0
    assert config.return_diagnostics is True


def test_app_loads_model_once_and_reports_active_backend():
    load_calls = []

    def load_model(model_dir, config):
        load_calls.append((model_dir, config))
        return FakeLid()

    settings = ServerSettings(
        model_dir="/models/FireRedLID",
        backend="compile",
        use_gpu=False,
    )
    app = create_app(settings, model_loader=load_model)

    with TestClient(app) as client:
        first = client.get("/healthz")
        second = client.get("/healthz")

    assert first.status_code == 200
    assert first.json() == {
        "status": "ok",
        "backend": "compile",
        "dtype": "mixed",
        "encoder_dtype": "float16",
        "decoder_dtype": "float32",
    }
    assert second.json() == first.json()
    assert len(load_calls) == 1
    assert load_calls[0][0] == "/models/FireRedLID"


def test_infer_decodes_ordered_logical_batch_and_returns_results():
    class RecordingLid:
        active_backend = "eager"

        def __init__(self):
            self.calls = []

        def process(self, uttids, wav_inputs):
            self.calls.append((uttids, wav_inputs))
            return [
                {
                    "uttid": uttid,
                    "lang": f"lang-{index}",
                    "confidence": 0.9,
                }
                for index, uttid in enumerate(uttids)
            ]

    model = RecordingLid()
    app = create_app(
        ServerSettings(model_dir="/model", use_gpu=False),
        model_loader=lambda *_: model,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/lid",
            json={
                "inputs": [
                    {"uttid": "first", "audio_base64": encode_wav()},
                    {"uttid": "second", "audio_base64": encode_wav()},
                ]
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "backend": "eager",
        "dtype": "mixed",
        "encoder_dtype": "float16",
        "decoder_dtype": "float32",
        "results": [
            {"uttid": "first", "lang": "lang-0", "confidence": 0.9},
            {"uttid": "second", "lang": "lang-1", "confidence": 0.9},
        ],
    }
    uttids, wav_inputs = model.calls[0]
    assert len(uttids) == 2
    assert len(set(uttids)) == 2
    assert [sample_rate for sample_rate, _ in wav_inputs] == [16000, 16000]
    assert all(waveform.dtype == np.int16 for _, waveform in wav_inputs)
    assert all(waveform.ndim == 1 for _, waveform in wav_inputs)


def test_infer_rejects_invalid_base64_audio():
    app = create_app(
        ServerSettings(model_dir="/model", use_gpu=False),
        model_loader=lambda *_: FakeLid(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/lid",
            json={"inputs": [{"uttid": "bad", "audio_base64": "%%%"}]},
        )
        recovery = client.post(
            "/v1/lid",
            json={
                "inputs": [
                    {"uttid": "good", "audio_base64": encode_wav()}
                ]
            },
        )

    assert response.status_code == 400
    assert "bad" in response.json()["detail"]
    assert recovery.status_code == 200


def test_infer_rejects_multi_channel_audio():
    app = create_app(
        ServerSettings(model_dir="/model", use_gpu=False),
        model_loader=lambda *_: FakeLid(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/lid",
            json={
                "inputs": [
                    {
                        "uttid": "stereo",
                        "audio_base64": encode_wav(channels=2),
                    }
                ]
            },
        )

    assert response.status_code == 400
    assert "mono" in response.json()["detail"]


def test_infer_rejects_non_16khz_audio():
    app = create_app(
        ServerSettings(model_dir="/model", use_gpu=False),
        model_loader=lambda *_: FakeLid(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/lid",
            json={
                "inputs": [
                    {
                        "uttid": "wrong-rate",
                        "audio_base64": encode_wav(sample_rate=8000),
                    }
                ]
            },
        )

    assert response.status_code == 400
    assert "16000" in response.json()["detail"]


def test_infer_rejects_audio_shorter_than_one_fbank_frame():
    class RecordingLid(FakeLid):
        def __init__(self):
            self.calls = []

        def process(self, uttids, wav_inputs):
            self.calls.append(list(uttids))
            return super().process(uttids, wav_inputs)

    model = RecordingLid()
    app = create_app(
        ServerSettings(model_dir="/model", use_gpu=False),
        model_loader=lambda *_: model,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/lid",
            json={
                "inputs": [
                    {
                        "uttid": "too-short",
                        "audio_base64": encode_wav(sample_count=399),
                    }
                ]
            },
        )

    assert response.status_code == 400
    assert "25 ms" in response.json()["detail"]
    assert model.calls == []


def test_infer_rejects_audio_longer_than_server_limit():
    app = create_app(
        ServerSettings(
            model_dir="/model",
            use_gpu=False,
            max_audio_seconds=0.05,
        ),
        model_loader=lambda *_: FakeLid(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/lid",
            json={
                "inputs": [
                    {"uttid": "too-long", "audio_base64": encode_wav()}
                ]
            },
        )

    assert response.status_code == 413
    assert "0.05" in response.json()["detail"]


def test_infer_rejects_oversized_request_body_before_json_parsing():
    settings = ServerSettings(
        model_dir="/model",
        use_gpu=False,
        max_audio_seconds=0.001,
        max_request_batch_size=1,
    )
    app = create_app(settings, model_loader=lambda *_: FakeLid())

    with TestClient(app) as client:
        response = client.post(
            "/v1/lid",
            content=b"x" * (settings.max_request_body_bytes + 1),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert "request body" in response.json()["detail"]


def test_infer_rejects_request_over_batch_limit():
    app = create_app(
        ServerSettings(
            model_dir="/model",
            use_gpu=False,
            max_request_batch_size=1,
        ),
        model_loader=lambda *_: FakeLid(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/lid",
            json={
                "inputs": [
                    {"uttid": "one", "audio_base64": encode_wav()},
                    {"uttid": "two", "audio_base64": encode_wav()},
                ]
            },
        )

    assert response.status_code == 413


def test_infer_returns_per_item_error_without_leaking_runtime_detail(caplog):
    class EmptyLid(FakeLid):
        def process(self, uttids, wav_inputs):
            return []

    app = create_app(
        ServerSettings(model_dir="/model", use_gpu=False),
        model_loader=lambda *_: EmptyLid(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/lid",
            json={
                "inputs": [
                    {"uttid": "missing", "audio_base64": encode_wav()}
                ]
            },
        )

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["errors"] == [
        {"uttid": "missing", "code": "inference_failed"}
    ]
    assert "missing result" not in response.text
    assert "FireRedLID item inference failed" in caplog.text


def test_infer_preserves_successes_when_one_item_fails():
    class PartialLid(FakeLid):
        active_backend = "eager"

        def process(self, uttids, wav_inputs):
            return [
                {
                    "uttid": uttids[1],
                    "lang": "en",
                    "confidence": 0.9,
                }
            ]

    app = create_app(
        ServerSettings(
            model_dir="/model",
            use_gpu=False,
            max_batch_delay_ms=0,
        ),
        model_loader=lambda *_: PartialLid(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/lid",
            json={
                "inputs": [
                    {"uttid": "failed", "audio_base64": encode_wav()},
                    {"uttid": "success", "audio_base64": encode_wav()},
                ]
            },
        )

    assert response.status_code == 200
    assert [result["uttid"] for result in response.json()["results"]] == [
        "success"
    ]
    assert response.json()["errors"] == [
        {"uttid": "failed", "code": "inference_failed"}
    ]


def test_unexpected_http_500_uses_generic_detail():
    app = create_app(
        ServerSettings(model_dir="/model", use_gpu=False),
        model_loader=lambda *_: FakeLid(),
    )

    async def broken_predict(inputs):
        raise RuntimeError("secret backend path")

    with TestClient(app) as client:
        client.app.state.lid_service.predict = broken_predict
        response = client.post(
            "/v1/lid",
            json={
                "inputs": [
                    {"uttid": "one", "audio_base64": encode_wav()}
                ]
            },
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "internal inference error"
    assert "secret backend path" not in response.text


def test_scheduler_closed_maps_to_http_503():
    app = create_app(
        ServerSettings(model_dir="/model", use_gpu=False),
        model_loader=lambda *_: FakeLid(),
    )

    async def closed_predict(inputs):
        raise server_module.SchedulerClosedError("scheduler closed")

    with TestClient(app) as client:
        client.app.state.lid_service.predict = closed_predict
        response = client.post(
            "/v1/lid",
            json={
                "inputs": [
                    {"uttid": "one", "audio_base64": encode_wav()}
                ]
            },
        )

    assert response.status_code == 503


def test_model_load_and_inference_use_the_same_dedicated_thread():
    thread_ids = {}

    class ThreadRecordingLid(FakeLid):
        active_backend = "eager"

        def process(self, uttids, wav_inputs):
            thread_ids["infer"] = threading.get_ident()
            return super().process(uttids, wav_inputs)

    def load_model(*args):
        thread_ids["load"] = threading.get_ident()
        return ThreadRecordingLid()

    main_thread = threading.get_ident()
    app = create_app(
        ServerSettings(
            model_dir="/model",
            use_gpu=False,
            encoder_precision="fp32",
            decoder_precision="fp32",
            max_batch_delay_ms=0,
        ),
        model_loader=load_model,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/lid",
            json={
                "inputs": [
                    {"uttid": "one", "audio_base64": encode_wav()}
                ]
            },
        )

    assert response.status_code == 200
    assert thread_ids["load"] == thread_ids["infer"]
    assert thread_ids["load"] != main_thread


def test_health_remains_responsive_while_inference_thread_is_busy():
    started = threading.Event()
    release = threading.Event()

    class BlockingLid(FakeLid):
        active_backend = "eager"

        def process(self, uttids, wav_inputs):
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release inference")
            return super().process(uttids, wav_inputs)

    app = create_app(
        ServerSettings(
            model_dir="/model",
            use_gpu=False,
            max_batch_delay_ms=0,
        ),
        model_loader=lambda *_: BlockingLid(),
    )

    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=1) as executor:
            inference = executor.submit(
                client.post,
                "/v1/lid",
                json={
                    "inputs": [
                        {
                            "uttid": "blocked",
                            "audio_base64": encode_wav(),
                        }
                    ]
                },
            )
            assert started.wait(timeout=5)
            health = client.get("/healthz")
            release.set()
            response = inference.result(timeout=5)

    assert health.status_code == 200
    assert response.status_code == 200


def test_health_returns_503_when_scheduler_is_unhealthy():
    app = create_app(
        ServerSettings(model_dir="/model", use_gpu=False),
        model_loader=lambda *_: FakeLid(),
    )

    with TestClient(app) as client:
        scheduler = client.app.state.lid_service._scheduler
        scheduler._fatal_error = ValueError("worker failed")
        health = client.get("/healthz")

    assert health.status_code == 503


def test_liveness_requests_restart_after_unrecoverable_scheduler_failure():
    app = create_app(
        ServerSettings(model_dir="/model", use_gpu=False),
        model_loader=lambda *_: FakeLid(),
    )

    with TestClient(app) as client:
        scheduler = client.app.state.lid_service._scheduler
        scheduler._fatal_error = ValueError("worker failed")
        liveness = client.get("/livez")
        readiness = client.get("/readyz")

    assert liveness.status_code == 503
    assert readiness.status_code == 503


def test_lifespan_releases_executors_when_scheduler_stop_raises(monkeypatch):
    real_executor = server_module.ThreadPoolExecutor
    shutdown_threads = []

    class RecordingExecutor(real_executor):
        def shutdown(self, *args, **kwargs):
            shutdown_threads.append(self._thread_name_prefix)
            return super().shutdown(*args, **kwargs)

    class BrokenStopScheduler(server_module.LidBatchScheduler):
        async def stop(self):
            await super().stop()
            raise RuntimeError("scheduler stop failed")

    monkeypatch.setattr(
        server_module,
        "ThreadPoolExecutor",
        RecordingExecutor,
    )
    monkeypatch.setattr(
        server_module,
        "LidBatchScheduler",
        BrokenStopScheduler,
    )
    app = create_app(
        ServerSettings(model_dir="/model", use_gpu=False),
        model_loader=lambda *_: FakeLid(),
    )

    with pytest.raises(RuntimeError, match="scheduler stop failed"):
        with TestClient(app):
            pass

    assert shutdown_threads == ["lid-decode", "lid-gpu"]


def test_server_settings_reject_non_positive_request_batch_limit():
    with pytest.raises(ValueError, match="max_request_batch_size"):
        ServerSettings(
            model_dir="/model",
            max_request_batch_size=0,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"queue_capacity": 0}, "queue_capacity"),
        ({"decode_workers": 0}, "decode_workers"),
        ({"max_batch_delay_ms": -1}, "max_batch_delay_ms"),
        ({"encoder_precision": "int8"}, "encoder_precision"),
        ({"decoder_precision": "int8"}, "decoder_precision"),
    ],
)
def test_server_settings_reject_invalid_scheduler_and_precision_values(
    overrides,
    message,
):
    with pytest.raises(ValueError, match=message):
        ServerSettings(model_dir="/model", **overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"queue_capacity": True}, "queue_capacity"),
        ({"max_batch_delay_ms": float("nan")}, "max_batch_delay_ms"),
        ({"max_audio_seconds": float("inf")}, "max_audio_seconds"),
        (
            {"bucket_policies": (BucketPolicy(float("nan"), 1),)},
            "bucket_policies",
        ),
    ],
)
def test_server_settings_reject_boolean_and_non_finite_numbers(
    overrides,
    message,
):
    with pytest.raises(ValueError, match=message):
        ServerSettings(model_dir="/model", **overrides)


def test_server_settings_require_ordered_buckets_covering_audio_limit():
    with pytest.raises(ValueError, match="bucket_policies"):
        ServerSettings(
            model_dir="/model",
            bucket_policies=(
                BucketPolicy(30, 4),
                BucketPolicy(5, 8),
            ),
        )
    with pytest.raises(ValueError, match="max_audio_seconds"):
        ServerSettings(
            model_dir="/model",
            max_audio_seconds=60,
            bucket_policies=(BucketPolicy(30, 4),),
        )


def test_parse_settings_supports_tensorrt_startup_options():
    settings = parse_settings(
        [
            "--model-dir",
            "/models/FireRedLID",
            "--backend",
            "tensorrt",
            "--profile",
            "throughput",
            "--use-half",
            "--engine-dir",
            "/engines/l20",
            "--fallback-backend",
            "eager",
            "--max-request-batch-size",
            "8",
            "--port",
            "9000",
        ]
    )

    assert settings.model_dir == "/models/FireRedLID"
    assert settings.backend == "tensorrt"
    assert settings.profile == "throughput"
    assert settings.use_gpu is True
    assert settings.use_half is True
    assert settings.encoder_precision == "fp16"
    assert settings.decoder_precision == "fp16"
    assert settings.engine_dir == "/engines/l20"
    assert settings.fallback_backend == "eager"
    assert settings.max_request_batch_size == 8
    assert settings.port == 9000


def test_legacy_use_half_rejects_explicit_precision_config(tmp_path):
    config_path = tmp_path / "server.yaml"
    config_path.write_text(
        """
runtime:
  encoder_precision: fp16
  decoder_precision: fp32
model:
  model_dir: /models/FireRedLID
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="use_half.*precision"):
        parse_settings(
            [
                "--config",
                str(config_path),
                "--use-half",
            ]
        )


def test_parse_settings_loads_yaml_and_applies_cli_overrides(tmp_path):
    config_path = tmp_path / "server.yaml"
    config_path.write_text(
        """
server:
  host: 127.0.0.1
  port: 7000
  max_request_batch_size: 12
  queue_capacity: 128
  decode_workers: 3
scheduler:
  max_batch_delay_ms: 9
  buckets:
    - {max_seconds: 10, max_batch_size: 6}
    - {max_seconds: 60, max_batch_size: 2}
runtime:
  backend: compile
  encoder_precision: bf16
  decoder_precision: fp32
model:
  model_dir: /models/from-yaml
  max_audio_seconds: 60
""",
        encoding="utf-8",
    )

    settings = parse_settings(
        [
            "--config",
            str(config_path),
            "--port",
            "9000",
            "--decoder-precision",
            "bf16",
        ]
    )

    assert settings.model_dir == "/models/from-yaml"
    assert settings.backend == "compile"
    assert settings.encoder_precision == "bf16"
    assert settings.decoder_precision == "bf16"
    assert settings.host == "127.0.0.1"
    assert settings.port == 9000
    assert settings.queue_capacity == 128
    assert settings.decode_workers == 3
    assert settings.max_batch_delay_ms == 9
    assert [
        (policy.max_seconds, policy.max_batch_size)
        for policy in settings.bucket_policies
    ] == [(10.0, 6), (60.0, 2)]


@pytest.mark.parametrize(
    "yaml_text",
    [
        "unknown_section: {}\nmodel:\n  model_dir: /model\n",
        "server:\n  backend: eager\nmodel:\n  model_dir: /model\n",
        "server:\n  mystery: 1\nmodel:\n  model_dir: /model\n",
    ],
)
def test_yaml_rejects_unknown_or_misplaced_keys(tmp_path, yaml_text):
    config_path = tmp_path / "server.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config key"):
        parse_settings(["--config", str(config_path)])


def test_yaml_rejects_boolean_bucket_numbers(tmp_path):
    config_path = tmp_path / "server.yaml"
    config_path.write_text(
        """
model:
  model_dir: /model
scheduler:
  buckets:
    - {max_seconds: true, max_batch_size: 1}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bucket.*max_seconds"):
        parse_settings(["--config", str(config_path)])


def test_checked_in_server_yaml_uses_production_defaults():
    settings = parse_settings(
        [
            "--config",
            "configs/fireredlid_server.yaml",
            "--model-dir",
            "/models/FireRedLID",
        ]
    )
    config = yaml.safe_load(
        Path("configs/fireredlid_server.yaml").read_text(encoding="utf-8")
    )

    assert "model_dir" not in config["model"]
    assert settings.model_dir == "/models/FireRedLID"
    assert settings.port == 12345
    assert settings.queue_capacity == 512
    assert settings.decode_workers == 8
    assert settings.encoder_precision == "fp16"
    assert settings.decoder_precision == "fp32"
    assert [
        (policy.max_seconds, policy.max_batch_size)
        for policy in settings.bucket_policies
    ] == [(5.0, 32), (15.0, 16), (30.0, 8), (60.0, 4)]


def test_main_starts_exactly_one_uvicorn_worker():
    calls = []

    def run_server(app, **kwargs):
        calls.append((app, kwargs))

    main(
        ["--model-dir", "/models/FireRedLID", "--no-use-gpu"],
        server_runner=run_server,
    )

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs == {
        "host": "0.0.0.0",
        "port": 12345,
        "log_level": "info",
        "workers": 1,
    }


def test_health_route_does_not_use_fastapi_threadpool():
    app = create_app(
        ServerSettings(model_dir="/model", use_gpu=False),
        model_loader=lambda *_: FakeLid(),
    )
    health_route = next(
        route for route in app.routes if route.path == "/healthz"
    )

    assert inspect.iscoroutinefunction(health_route.endpoint)


def test_server_dynamically_batches_concurrent_single_item_requests():
    class RecordingLid(FakeLid):
        active_backend = "eager"

        def __init__(self):
            self.calls = []
            self.lock = threading.Lock()

        def process(self, uttids, wav_inputs):
            with self.lock:
                self.calls.append(list(uttids))
            return super().process(uttids, wav_inputs)

    model = RecordingLid()
    app = create_app(
        ServerSettings(
            model_dir="/model",
            use_gpu=False,
            encoder_precision="fp32",
            decoder_precision="fp32",
            max_batch_delay_ms=50,
            bucket_policies=(BucketPolicy(60, 4),),
        ),
        model_loader=lambda *_: model,
    )

    with TestClient(app) as client:
        barrier = threading.Barrier(4)

        def send(index):
            barrier.wait()
            return client.post(
                "/v1/lid",
                json={
                    "inputs": [
                        {
                            "uttid": f"utt-{index}",
                            "audio_base64": encode_wav(),
                        }
                    ]
                },
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(send, index) for index in range(3)]
            barrier.wait()
            responses = [future.result() for future in futures]

    assert all(response.status_code == 200 for response in responses)
    assert len(model.calls) == 1
    assert len(model.calls[0]) == 3
    assert {
        response.json()["results"][0]["uttid"]
        for response in responses
    } == {"utt-0", "utt-1", "utt-2"}


def test_server_settings_reject_queue_smaller_than_request_limit():
    with pytest.raises(ValueError, match="queue_capacity"):
        ServerSettings(
            model_dir="/model",
            use_gpu=False,
            encoder_precision="fp32",
            decoder_precision="fp32",
            queue_capacity=1,
            max_request_batch_size=2,
        )


def test_queue_admission_happens_before_audio_decode(monkeypatch):
    def unexpected_decode(*args, **kwargs):
        raise AssertionError("audio decode must not run for a rejected request")

    monkeypatch.setattr(server_module, "decode_audio", unexpected_decode)
    app = create_app(
        ServerSettings(
            model_dir="/model",
            use_gpu=False,
            queue_capacity=2,
            max_request_batch_size=2,
        ),
        model_loader=lambda *_: FakeLid(),
    )

    with TestClient(app) as client:
        def reject_reservation(count):
            raise server_module.QueueFullError("inference queue is full")

        monkeypatch.setattr(
            client.app.state.lid_service._scheduler,
            "reserve",
            reject_reservation,
        )
        response = client.post(
            "/v1/lid",
            json={
                "inputs": [
                    {"uttid": "one", "audio_base64": "unused"},
                    {"uttid": "two", "audio_base64": "unused"},
                ]
            },
        )

    assert response.status_code == 429
    assert response.json()["detail"] == "inference queue is full"


def test_scheduler_caps_physical_batch_to_backend_limit():
    class LimitedLid(FakeLid):
        active_backend = "tensorrt"
        backend_max_batch = 2

        def __init__(self):
            self.calls = []

        def process(self, uttids, wav_inputs):
            self.calls.append(list(uttids))
            return super().process(uttids, wav_inputs)

    model = LimitedLid()
    app = create_app(
        ServerSettings(
            model_dir="/model",
            use_gpu=False,
            max_batch_delay_ms=0,
            bucket_policies=(BucketPolicy(60, 8),),
        ),
        model_loader=lambda *_: model,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/lid",
            json={
                "inputs": [
                    {
                        "uttid": f"utt-{index}",
                        "audio_base64": encode_wav(),
                    }
                    for index in range(3)
                ]
            },
        )

    assert response.status_code == 200
    assert [len(call) for call in model.calls] == [2, 1]


def test_server_rejects_non_positive_backend_batch_limit():
    class InvalidBackendLimitLid(FakeLid):
        backend_max_batch = 0

    app = create_app(
        ServerSettings(model_dir="/model", use_gpu=False),
        model_loader=lambda *_: InvalidBackendLimitLid(),
    )

    with pytest.raises(ValueError, match="backend_max_batch.*positive"):
        with TestClient(app):
            pass


def test_pyproject_exposes_fireredlid_server_script():
    project = tomllib.loads(
        Path("pyproject.toml").read_text(encoding="utf-8")
    )

    assert project["project"]["scripts"]["fireredlid-server"] == (
        "fireredasr2s.fireredlid.server:main"
    )


def test_container_entrypoint_uses_checked_in_server_config():
    entrypoint = Path("entrypoint.sh").read_text(encoding="utf-8")

    assert "configs/fireredlid_server.yaml" in entrypoint
    assert "--use-half" not in entrypoint
