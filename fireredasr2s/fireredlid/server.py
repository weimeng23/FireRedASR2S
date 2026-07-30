import argparse
import asyncio
import base64
import binascii
import io
import logging
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import yaml
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .lid import FireRedLid, FireRedLidConfig
from .scheduler import (
    BucketPolicy,
    LidBatchScheduler,
    PreparedLidInput,
    QueueFullError,
    SchedulerClosedError,
)


logger = logging.getLogger(__name__)

MIN_AUDIO_DURATION_S = 0.025


@dataclass(frozen=True)
class ServerSettings:
    model_dir: str
    backend: str = "eager"
    profile: str = "latency"
    use_gpu: bool = True
    use_half: bool | None = None
    engine_dir: str | None = None
    fallback_backend: str | None = None
    max_audio_seconds: float = 60.0
    max_request_batch_size: int = 32
    queue_capacity: int = 512
    decode_workers: int = 16
    max_batch_delay_ms: float = 5.0
    bucket_policies: tuple[BucketPolicy, ...] = (
        BucketPolicy(5.0, 32),
        BucketPolicy(15.0, 16),
        BucketPolicy(30.0, 8),
        BucketPolicy(60.0, 4),
    )
    encoder_precision: str | None = None
    decoder_precision: str | None = None
    host: str = "0.0.0.0"
    port: int = 12345
    log_level: str = "info"

    def __post_init__(self):
        if self.use_half is not None:
            if (
                self.encoder_precision is not None
                or self.decoder_precision is not None
            ):
                raise ValueError(
                    "use_half conflicts with explicit precision settings"
                )
            precision = "fp16" if self.use_half else "fp32"
            object.__setattr__(self, "encoder_precision", precision)
            object.__setattr__(self, "decoder_precision", precision)
        else:
            object.__setattr__(
                self,
                "encoder_precision",
                self.encoder_precision or "fp16",
            )
            object.__setattr__(
                self,
                "decoder_precision",
                self.decoder_precision or "fp32",
            )
        if (
            isinstance(self.max_request_batch_size, bool)
            or not isinstance(self.max_request_batch_size, int)
            or self.max_request_batch_size <= 0
        ):
            raise ValueError("max_request_batch_size must be positive")
        if (
            isinstance(self.queue_capacity, bool)
            or not isinstance(self.queue_capacity, int)
            or self.queue_capacity <= 0
        ):
            raise ValueError("queue_capacity must be positive")
        if self.queue_capacity < self.max_request_batch_size:
            raise ValueError(
                "queue_capacity must be at least max_request_batch_size"
            )
        if (
            isinstance(self.decode_workers, bool)
            or not isinstance(self.decode_workers, int)
            or self.decode_workers <= 0
        ):
            raise ValueError("decode_workers must be positive")
        if (
            isinstance(self.max_batch_delay_ms, bool)
            or not isinstance(self.max_batch_delay_ms, (int, float))
            or not math.isfinite(self.max_batch_delay_ms)
            or self.max_batch_delay_ms < 0
        ):
            raise ValueError("max_batch_delay_ms must not be negative")
        if self.encoder_precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("unsupported encoder_precision")
        if self.decoder_precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("unsupported decoder_precision")
        if (
            isinstance(self.max_audio_seconds, bool)
            or not isinstance(self.max_audio_seconds, (int, float))
            or not math.isfinite(self.max_audio_seconds)
            or self.max_audio_seconds <= 0
        ):
            raise ValueError("max_audio_seconds must be positive")
        boundaries = [
            policy.max_seconds for policy in self.bucket_policies
        ]
        if (
            not boundaries
            or any(
                isinstance(policy.max_seconds, bool)
                or not isinstance(policy.max_seconds, (int, float))
                or not math.isfinite(policy.max_seconds)
                or policy.max_seconds <= 0
                or isinstance(policy.max_batch_size, bool)
                or not isinstance(policy.max_batch_size, int)
                or policy.max_batch_size <= 0
                for policy in self.bucket_policies
            )
            or any(
                current <= previous
                for previous, current in zip(
                    boundaries,
                    boundaries[1:],
                )
            )
        ):
            raise ValueError(
                "bucket_policies must be positive and strictly ordered"
            )
        if boundaries[-1] < self.max_audio_seconds:
            raise ValueError(
                "bucket_policies must cover max_audio_seconds"
            )
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
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
        if self.encoder_precision == self.decoder_precision:
            return {
                "fp32": "float32",
                "fp16": "float16",
                "bf16": "bfloat16",
            }[self.encoder_precision]
        return "mixed"

    @property
    def encoder_dtype(self):
        return {
            "fp32": "float32",
            "fp16": "float16",
            "bf16": "bfloat16",
        }[self.encoder_precision]

    @property
    def decoder_dtype(self):
        return {
            "fp32": "float32",
            "fp16": "float16",
            "bf16": "bfloat16",
        }[self.decoder_precision]


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
    min_samples = math.ceil(sample_rate * MIN_AUDIO_DURATION_S)
    if waveform.size < min_samples:
        raise InvalidAudioError(
            f"audio for uttid '{item.uttid}' must be at least 25 ms"
        )
    return sample_rate, waveform


def prepare_audio(item, max_audio_seconds, feature_extractor):
    wav_input = decode_audio(item, max_audio_seconds)
    feature_item = feature_extractor.extract_one(
        wav_input,
        item.uttid,
        max_audio_seconds,
    )
    if feature_item is None:
        raise InvalidAudioError(
            f"audio for uttid '{item.uttid}' produced no feature frames"
        )
    return PreparedLidInput(
        uttid=item.uttid,
        feature=feature_item.feature,
        duration_s=feature_item.duration_s,
        processed_duration_s=feature_item.processed_duration_s,
        truncated=feature_item.truncated,
    )


class LidService:
    def __init__(
        self,
        model,
        scheduler,
        preprocess_executor,
        max_request_batch_size: int,
        max_audio_seconds: float = 60.0,
    ):
        self._model = model
        self._scheduler = scheduler
        self._preprocess_executor = preprocess_executor
        self._feature_extractor = model.feat_extractor
        self._max_request_batch_size = max_request_batch_size
        self._max_audio_seconds = max_audio_seconds

    @property
    def active_backend(self):
        return self._model.active_backend

    @property
    def pending_count(self):
        return self._scheduler.pending_count

    @property
    def is_healthy(self):
        return self._scheduler.is_healthy

    async def predict(self, inputs: list[LidInput]):
        if len(inputs) > self._max_request_batch_size:
            raise RequestBatchTooLarge(
                "request contains "
                f"{len(inputs)} inputs; maximum is "
                f"{self._max_request_batch_size}"
            )
        reservation = self._scheduler.reserve(len(inputs))
        loop = asyncio.get_running_loop()
        try:
            prepared_outputs = await asyncio.gather(
                *[
                    loop.run_in_executor(
                        self._preprocess_executor,
                        prepare_audio,
                        item,
                        self._max_audio_seconds,
                        self._feature_extractor,
                    )
                    for item in inputs
                ],
                return_exceptions=True,
            )
            for output in prepared_outputs:
                if isinstance(output, BaseException):
                    raise output
            futures = self._scheduler.submit_reserved(
                reservation,
                prepared_outputs,
            )
        except BaseException:
            self._scheduler.release(reservation)
            raise
        outcomes = await asyncio.gather(
            *futures,
            return_exceptions=True,
        )
        results = []
        errors = []
        for item, outcome in zip(inputs, outcomes, strict=True):
            if isinstance(outcome, SchedulerClosedError):
                raise outcome
            if isinstance(outcome, BaseException):
                logger.error(
                    "FireRedLID item inference failed for uttid=%r",
                    item.uttid,
                    exc_info=(
                        type(outcome),
                        outcome,
                        outcome.__traceback__,
                    ),
                )
                errors.append(
                    {
                        "uttid": item.uttid,
                        "code": "inference_failed",
                    }
                )
                continue
            results.append(outcome)
        return results, errors


def create_lid_config(settings: ServerSettings) -> FireRedLidConfig:
    return FireRedLidConfig(
        use_gpu=settings.use_gpu,
        encoder_precision=settings.encoder_precision,
        decoder_precision=settings.decoder_precision,
        backend=settings.backend,
        profile=settings.profile,
        max_audio_seconds=settings.max_audio_seconds,
        engine_dir=settings.engine_dir,
        fallback_backend=settings.fallback_backend,
        return_diagnostics=True,
    )


def create_app(
    settings: ServerSettings,
    model_loader: Callable = FireRedLid.from_pretrained,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        preprocess_executor = ThreadPoolExecutor(
            max_workers=settings.decode_workers,
            thread_name_prefix="lid-preprocess",
        )
        gpu_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="lid-gpu",
        )
        scheduler = None
        config = create_lid_config(settings)
        loop = asyncio.get_running_loop()
        try:
            model = await loop.run_in_executor(
                gpu_executor,
                model_loader,
                settings.model_dir,
                config,
            )
            backend_max_batch = getattr(
                model,
                "backend_max_batch",
                None,
            )
            if (
                backend_max_batch is not None
                and backend_max_batch <= 0
            ):
                raise ValueError("backend_max_batch must be positive")
            bucket_policies = tuple(
                BucketPolicy(
                    policy.max_seconds,
                    min(policy.max_batch_size, backend_max_batch)
                    if backend_max_batch is not None
                    else policy.max_batch_size,
                )
                for policy in settings.bucket_policies
            )
            scheduler = LidBatchScheduler(
                engine=model,
                executor=gpu_executor,
                queue_capacity=settings.queue_capacity,
                max_batch_delay_ms=settings.max_batch_delay_ms,
                bucket_policies=bucket_policies,
            )
            await scheduler.start()
            app.state.lid_service = LidService(
                model,
                scheduler,
                preprocess_executor,
                max_request_batch_size=settings.max_request_batch_size,
                max_audio_seconds=settings.max_audio_seconds,
            )
            yield
        finally:
            try:
                if scheduler is not None:
                    await scheduler.stop()
            finally:
                if hasattr(app.state, "lid_service"):
                    del app.state.lid_service
                try:
                    preprocess_executor.shutdown(wait=True)
                finally:
                    gpu_executor.shutdown(wait=True)

    app = FastAPI(title="FireRedLID Server", lifespan=lifespan)

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

    @app.get("/livez")
    async def liveness(request: Request):
        service = request.app.state.lid_service
        if not service.is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="inference scheduler requires process restart",
            )
        return {"status": "ok"}

    @app.get("/healthz")
    @app.get("/readyz")
    async def readiness(request: Request):
        service = request.app.state.lid_service
        if not service.is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="inference scheduler is not healthy",
            )
        return {
            "status": "ok",
            "backend": service.active_backend,
            "dtype": settings.dtype,
            "encoder_dtype": settings.encoder_dtype,
            "decoder_dtype": settings.decoder_dtype,
        }

    @app.post("/v1/lid")
    async def infer(payload: LidRequest, request: Request):
        service = request.app.state.lid_service
        try:
            results, errors = await service.predict(payload.inputs)
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
        except QueueFullError as error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(error),
            ) from error
        except SchedulerClosedError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except Exception as error:
            logger.exception("FireRedLID inference failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="internal inference error",
            ) from error
        response = {
            "backend": service.active_backend,
            "dtype": settings.dtype,
            "encoder_dtype": settings.encoder_dtype,
            "decoder_dtype": settings.decoder_dtype,
            "results": results,
        }
        if errors:
            response["errors"] = errors
        return response

    return app


def _load_yaml_settings(path: str) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("server config must be a YAML mapping")
    section_keys = {
        "server": {
            "host",
            "port",
            "log_level",
            "max_request_batch_size",
            "queue_capacity",
            "decode_workers",
        },
        "scheduler": {
            "max_batch_delay_ms",
            "buckets",
        },
        "runtime": {
            "backend",
            "profile",
            "use_gpu",
            "use_half",
            "encoder_precision",
            "decoder_precision",
            "engine_dir",
            "fallback_backend",
        },
        "model": {
            "model_dir",
            "max_audio_seconds",
        },
    }
    unknown_sections = set(data) - set(section_keys)
    if unknown_sections:
        name = sorted(unknown_sections)[0]
        raise ValueError(f"unknown config key: {name}")
    values = {}
    for section_name in ("server", "runtime", "model"):
        section = data.get(section_name, {})
        if not isinstance(section, dict):
            raise ValueError(f"{section_name} config must be a mapping")
        unknown_keys = set(section) - section_keys[section_name]
        if unknown_keys:
            key = sorted(unknown_keys)[0]
            raise ValueError(
                f"unknown config key: {section_name}.{key}"
            )
        values.update(section)
    scheduler = data.get("scheduler", {})
    if not isinstance(scheduler, dict):
        raise ValueError("scheduler config must be a mapping")
    unknown_keys = set(scheduler) - section_keys["scheduler"]
    if unknown_keys:
        key = sorted(unknown_keys)[0]
        raise ValueError(f"unknown config key: scheduler.{key}")
    values.update(
        {
            key: value
            for key, value in scheduler.items()
            if key != "buckets"
        }
    )
    buckets = scheduler.get("buckets")
    if buckets is not None:
        if not isinstance(buckets, list):
            raise ValueError("scheduler buckets must be a list")
        policies = []
        for index, bucket in enumerate(buckets):
            if not isinstance(bucket, dict):
                raise ValueError(
                    f"scheduler bucket {index} must be a mapping"
                )
            if set(bucket) != {"max_seconds", "max_batch_size"}:
                raise ValueError(
                    f"scheduler bucket {index} must contain only "
                    "max_seconds and max_batch_size"
                )
            max_seconds = bucket["max_seconds"]
            max_batch_size = bucket["max_batch_size"]
            if (
                isinstance(max_seconds, bool)
                or not isinstance(max_seconds, (int, float))
                or not math.isfinite(max_seconds)
            ):
                raise ValueError(
                    f"scheduler bucket {index} max_seconds "
                    "must be a finite number"
                )
            if (
                isinstance(max_batch_size, bool)
                or not isinstance(max_batch_size, int)
            ):
                raise ValueError(
                    f"scheduler bucket {index} max_batch_size "
                    "must be an integer"
                )
            policies.append(
                BucketPolicy(
                    max_seconds=float(max_seconds),
                    max_batch_size=max_batch_size,
                )
            )
        values["bucket_policies"] = tuple(policies)
    return values


def parse_settings(argv=None) -> ServerSettings:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args(argv)
    values = (
        _load_yaml_settings(config_args.config)
        if config_args.config is not None
        else {}
    )

    parser = argparse.ArgumentParser(
        description="Serve FireRedLID through FastAPI",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--config")
    parser.add_argument("--model-dir")
    parser.add_argument(
        "--backend",
        choices=("eager", "compile", "tensorrt"),
    )
    parser.add_argument(
        "--profile",
        choices=("latency", "throughput"),
    )
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--use-half",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--encoder-precision",
        choices=("fp32", "fp16", "bf16"),
    )
    parser.add_argument(
        "--decoder-precision",
        choices=("fp32", "fp16", "bf16"),
    )
    parser.add_argument("--engine-dir")
    parser.add_argument("--fallback-backend", choices=("eager",))
    parser.add_argument("--max-audio-seconds", type=float)
    parser.add_argument("--max-request-batch-size", type=int)
    parser.add_argument("--queue-capacity", type=int)
    parser.add_argument("--decode-workers", type=int)
    parser.add_argument("--max-batch-delay-ms", type=float)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--log-level")
    args = parser.parse_args(argv)
    overrides = vars(args)
    overrides.pop("config", None)
    values.update(overrides)
    if "model_dir" not in values:
        parser.error("--model-dir is required unless supplied by --config")
    return ServerSettings(**values)


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
