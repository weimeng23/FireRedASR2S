import asyncio
import base64
import io
import inspect
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import fireredasr2s.fireredlid.server as server_module
from fireredasr2s.fireredlid.server import (
    LidInput,
    LidRequest,
    LidService,
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


def encode_wav(*, channels=1, sample_rate=16000):
    samples = np.arange(1600, dtype=np.int16)
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


def test_server_settings_preserve_runtime_backend_configuration():
    settings = ServerSettings(
        model_dir="/models/FireRedLID",
        backend="tensorrt",
        profile="throughput",
        use_gpu=True,
        use_half=True,
        engine_dir="/engines/l20",
        fallback_backend="eager",
        max_sub_batch_size=4,
        max_audio_seconds=30.0,
    )

    config = create_lid_config(settings)

    assert config.backend == "tensorrt"
    assert config.profile == "throughput"
    assert config.use_gpu is True
    assert config.use_half is True
    assert config.engine_dir == "/engines/l20"
    assert config.fallback_backend == "eager"
    assert config.max_sub_batch_size == 4
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
    assert first.json() == {"status": "ok", "backend": "compile"}
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
        "results": [
            {"uttid": "first", "lang": "lang-0", "confidence": 0.9},
            {"uttid": "second", "lang": "lang-1", "confidence": 0.9},
        ],
    }
    uttids, wav_inputs = model.calls[0]
    assert uttids == ["first", "second"]
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

    assert response.status_code == 400
    assert "bad" in response.json()["detail"]


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


def test_infer_returns_500_when_runtime_drops_results():
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

    assert response.status_code == 500
    assert "result count" in response.json()["detail"]


def test_lid_service_serializes_calls_to_shared_model():
    class SlowLid(FakeLid):
        active_backend = "eager"

        def __init__(self):
            self.inflight = 0
            self.max_inflight = 0
            self.state_lock = threading.Lock()

        def process(self, uttids, wav_inputs):
            with self.state_lock:
                self.inflight += 1
                self.max_inflight = max(self.max_inflight, self.inflight)
            time.sleep(0.05)
            with self.state_lock:
                self.inflight -= 1
            return super().process(uttids, wav_inputs)

    model = SlowLid()
    service = LidService(model, max_request_batch_size=1)
    item = LidInput(uttid="one", audio_base64=encode_wav())
    start = threading.Barrier(3)

    def predict():
        start.wait()
        return service.predict([item])

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(predict) for _ in range(2)]
        start.wait()
        results = [future.result() for future in futures]

    assert len(results) == 2
    assert model.max_inflight == 1


def test_server_settings_reject_non_positive_request_batch_limit():
    with pytest.raises(ValueError, match="max_request_batch_size"):
        ServerSettings(
            model_dir="/model",
            max_request_batch_size=0,
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
            "--max-sub-batch-size",
            "4",
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
    assert settings.engine_dir == "/engines/l20"
    assert settings.fallback_backend == "eager"
    assert settings.max_sub_batch_size == 4
    assert settings.max_request_batch_size == 8
    assert settings.port == 9000


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
        "port": 8000,
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


def test_inference_gate_limits_threadpool_submission(monkeypatch):
    app = create_app(
        ServerSettings(model_dir="/model", use_gpu=False),
        model_loader=lambda *_: FakeLid(),
    )
    app.state.lid_service = LidService(
        FakeLid(),
        max_request_batch_size=1,
    )
    infer_route = next(
        route for route in app.routes if route.path == "/v1/lid"
    )
    payload = LidRequest(
        inputs=[LidInput(uttid="one", audio_base64=encode_wav())]
    )
    request = SimpleNamespace(app=app)
    inflight = 0
    max_inflight = 0

    async def recording_run(function, *args):
        nonlocal inflight, max_inflight
        inflight += 1
        max_inflight = max(max_inflight, inflight)
        await asyncio.sleep(0.02)
        result = function(*args)
        inflight -= 1
        return result

    monkeypatch.setattr(server_module, "run_in_threadpool", recording_run)

    async def run_requests():
        await asyncio.gather(
            infer_route.endpoint(payload, request),
            infer_route.endpoint(payload, request),
        )

    asyncio.run(run_requests())

    assert max_inflight == 1


def test_pyproject_exposes_fireredlid_server_script():
    project = tomllib.loads(
        Path("pyproject.toml").read_text(encoding="utf-8")
    )

    assert project["project"]["scripts"]["fireredlid-server"] == (
        "fireredasr2s.fireredlid.server:main"
    )
