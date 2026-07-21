import argparse
import asyncio
import base64
import binascii
import io
import math
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .lid import FireRedLid, FireRedLidConfig


@dataclass(frozen=True)
class ServerSettings:
    model_dir: str
    backend: str = "eager"
    profile: str = "latency"
    use_gpu: bool = True
    use_half: bool = False
    engine_dir: str | None = None
    fallback_backend: str | None = None
    max_sub_batch_size: int | None = None
    max_audio_seconds: float = 60.0
    max_request_batch_size: int = 32
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    def __post_init__(self):
        if self.max_request_batch_size <= 0:
            raise ValueError("max_request_batch_size must be positive")
        if self.max_sub_batch_size is not None and self.max_sub_batch_size <= 0:
            raise ValueError("max_sub_batch_size must be positive")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")

    @property
    def max_audio_file_bytes(self):
        pcm_bytes = math.ceil(self.max_audio_seconds * 16000) * 2
        return pcm_bytes + 64 * 1024

    @property
    def max_request_body_bytes(self):
        encoded_audio_bytes = 4 * math.ceil(self.max_audio_file_bytes / 3)
        per_item_json_bytes = encoded_audio_bytes + 1024
        return self.max_request_batch_size * per_item_json_bytes + 1024

    @property
    def dtype(self):
        return "float16" if self.use_half else "float32"


class LidInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uttid: str = Field(min_length=1)
    audio_base64: str = Field(min_length=1)


class LidRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: list[LidInput] = Field(min_length=1)


class InvalidAudioError(ValueError):
    pass


class AudioTooLargeError(ValueError):
    pass


class RequestBatchTooLarge(ValueError):
    pass


class InferenceResultError(RuntimeError):
    pass


def decode_audio(item: LidInput, max_audio_seconds: float):
    max_audio_file_bytes = (
        math.ceil(max_audio_seconds * 16000) * 2 + 64 * 1024
    )
    max_encoded_chars = 4 * math.ceil(max_audio_file_bytes / 3)
    if len(item.audio_base64) > max_encoded_chars:
        raise AudioTooLargeError(
            f"audio for uttid '{item.uttid}' exceeds "
            f"{max_audio_seconds:g} seconds or encoded size limit"
        )
    try:
        audio_bytes = base64.b64decode(item.audio_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise InvalidAudioError(
            f"invalid base64 audio for uttid '{item.uttid}'"
        ) from error

    if len(audio_bytes) > max_audio_file_bytes:
        raise AudioTooLargeError(
            f"audio for uttid '{item.uttid}' exceeds "
            f"{max_audio_seconds:g} seconds or encoded size limit"
        )

    try:
        audio_file = sf.SoundFile(io.BytesIO(audio_bytes))
    except (RuntimeError, TypeError, ValueError) as error:
        raise InvalidAudioError(
            f"unreadable audio for uttid '{item.uttid}'"
        ) from error

    with audio_file:
        if audio_file.channels != 1:
            raise InvalidAudioError(
                f"audio for uttid '{item.uttid}' must be mono"
            )
        if audio_file.samplerate != 16000:
            raise InvalidAudioError(
                f"audio for uttid '{item.uttid}' must use sample rate 16000"
            )
        max_frames = math.floor(max_audio_seconds * audio_file.samplerate)
        if audio_file.frames > max_frames:
            raise AudioTooLargeError(
                f"audio for uttid '{item.uttid}' exceeds "
                f"{max_audio_seconds:g} seconds"
            )
        try:
            waveform = audio_file.read(dtype="int16", always_2d=False)
        except (RuntimeError, TypeError, ValueError) as error:
            raise InvalidAudioError(
                f"unreadable audio for uttid '{item.uttid}'"
            ) from error

    if waveform.size == 0:
        raise InvalidAudioError(
            f"audio for uttid '{item.uttid}' must not be empty"
        )
    sample_rate = audio_file.samplerate
    return sample_rate, waveform


class LidService:
    def __init__(
        self,
        model,
        max_request_batch_size: int,
        max_audio_seconds: float = 60.0,
    ):
        self._model = model
        self._max_request_batch_size = max_request_batch_size
        self._max_audio_seconds = max_audio_seconds
        self._inference_lock = threading.Lock()

    @property
    def active_backend(self):
        return self._model.active_backend

    def predict(self, inputs: list[LidInput]):
        if len(inputs) > self._max_request_batch_size:
            raise RequestBatchTooLarge(
                "request contains "
                f"{len(inputs)} inputs; maximum is "
                f"{self._max_request_batch_size}"
            )
        wav_inputs = [
            decode_audio(item, self._max_audio_seconds) for item in inputs
        ]
        uttids = [item.uttid for item in inputs]
        with self._inference_lock:
            results = self._model.process(uttids, wav_inputs)
        if len(results) != len(inputs):
            raise InferenceResultError(
                "runtime result count does not match request input count"
            )
        return results


def create_lid_config(settings: ServerSettings) -> FireRedLidConfig:
    return FireRedLidConfig(
        use_gpu=settings.use_gpu,
        use_half=settings.use_half,
        backend=settings.backend,
        profile=settings.profile,
        max_audio_seconds=settings.max_audio_seconds,
        engine_dir=settings.engine_dir,
        max_sub_batch_size=settings.max_sub_batch_size,
        fallback_backend=settings.fallback_backend,
        return_diagnostics=True,
    )


def create_app(
    settings: ServerSettings,
    model_loader: Callable = FireRedLid.from_pretrained,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = create_lid_config(settings)
        model = model_loader(settings.model_dir, config)
        app.state.lid_service = LidService(
            model,
            max_request_batch_size=settings.max_request_batch_size,
            max_audio_seconds=settings.max_audio_seconds,
        )
        yield
        del app.state.lid_service

    app = FastAPI(title="FireRedLID Server", lifespan=lifespan)
    app.state.inference_gate = asyncio.Semaphore(1)

    @app.middleware("http")
    async def enforce_request_body_limit(request: Request, call_next):
        if request.url.path == "/v1/lid":
            content_length = request.headers.get("content-length")
            try:
                body_size = int(content_length) if content_length else None
            except ValueError:
                body_size = None
            if (
                body_size is not None
                and body_size > settings.max_request_body_bytes
            ):
                return JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={
                        "detail": "request body exceeds configured size limit"
                    },
                )
        return await call_next(request)

    @app.get("/healthz")
    async def health(request: Request):
        return {
            "status": "ok",
            "backend": request.app.state.lid_service.active_backend,
            "dtype": settings.dtype,
        }

    @app.post("/v1/lid")
    async def infer(payload: LidRequest, request: Request):
        service = request.app.state.lid_service
        try:
            async with request.app.state.inference_gate:
                results = await run_in_threadpool(
                    service.predict,
                    payload.inputs,
                )
        except InvalidAudioError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        except AudioTooLargeError as error:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(error),
            ) from error
        except RequestBatchTooLarge as error:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(error),
            ) from error
        except InferenceResultError as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(error),
            ) from error
        return {
            "backend": service.active_backend,
            "dtype": settings.dtype,
            "results": results,
        }

    return app


def parse_settings(argv=None) -> ServerSettings:
    parser = argparse.ArgumentParser(
        description="Serve FireRedLID through FastAPI",
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--backend",
        choices=("eager", "compile", "tensorrt"),
        default="eager",
    )
    parser.add_argument(
        "--profile",
        choices=("latency", "throughput"),
        default="latency",
    )
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-half",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--engine-dir")
    parser.add_argument("--fallback-backend", choices=("eager",))
    parser.add_argument("--max-sub-batch-size", type=int)
    parser.add_argument("--max-audio-seconds", type=float, default=60.0)
    parser.add_argument("--max-request-batch-size", type=int, default=32)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    return ServerSettings(
        model_dir=args.model_dir,
        backend=args.backend,
        profile=args.profile,
        use_gpu=args.use_gpu,
        use_half=args.use_half,
        engine_dir=args.engine_dir,
        fallback_backend=args.fallback_backend,
        max_sub_batch_size=args.max_sub_batch_size,
        max_audio_seconds=args.max_audio_seconds,
        max_request_batch_size=args.max_request_batch_size,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


def main(argv=None, server_runner=None):
    settings = parse_settings(argv)
    if server_runner is None:
        import uvicorn

        server_runner = uvicorn.run
    app = create_app(settings)
    server_runner(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        workers=1,
    )


if __name__ == "__main__":
    main()
