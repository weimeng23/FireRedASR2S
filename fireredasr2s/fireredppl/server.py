"""Raw int16 PCM /score API for speechflow's pcm_s16le client mode."""

import argparse
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, Request
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings
from .scheduler import PPLBatchScheduler, QueueFullError, SchedulerClosedError
from .scorer import InputError, InputTooLarge, PPLScorer


def prepare_pcm(engine, settings, uid, pcm_bytes, sample_rate, texts):
    if sample_rate != 16000:
        raise InputError("sample_rate must be 16000")
    if not pcm_bytes or len(pcm_bytes) % 2:
        raise InputError("PCM must be nonempty and contain complete int16 samples")
    if len(pcm_bytes) // 2 / sample_rate > settings.model.max_audio_seconds:
        raise InputTooLarge("audio exceeds max_audio_seconds; audio is not truncated")
    # Match the local Actor's int16 -> float32 conversion, without normalization.
    waveform = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
    return engine.prepare(uid, waveform, sample_rate, texts)


async def parse_request(request, settings):
    """Bound the entire body, including uploads, before multipart parsing."""
    limit = settings.server.max_request_bytes
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            length = int(content_length)
        except ValueError as error:
            raise HTTPException(400, "invalid Content-Length") from error
        if length < 0:
            raise HTTPException(400, "invalid Content-Length")
        if length > limit:
            raise InputTooLarge("request body exceeds max_request_bytes")
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "multipart/form-data":
        raise HTTPException(415, "expected multipart/form-data")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > limit:
            raise InputTooLarge("request body exceeds max_request_bytes")
        body.extend(chunk)

    async def receive():
        return {"type": "http.request", "body": bytes(body), "more_body": False}

    bounded = Request(request.scope, receive)
    async with bounded.form(max_files=1, max_fields=5, max_part_size=limit) as form:
        expected = {"pcm", "uid", "candidates", "sample_rate", "channels", "sample_format"}
        if set(form.keys()) != expected or len(form.multi_items()) != len(expected):
            raise HTTPException(422, "expected pcm, uid, candidates, sample_rate, channels, sample_format")
        upload, uid, raw_candidates = form["pcm"], form["uid"], form["candidates"]
        if not isinstance(upload, UploadFile):
            raise HTTPException(422, "pcm must be an uploaded binary int16 buffer")
        if upload.content_type != "application/octet-stream" or form["sample_format"] != "s16le":
            raise HTTPException(415, "expected application/octet-stream PCM with sample_format=s16le")
        if form["sample_rate"] != "16000" or form["channels"] != "1":
            raise InputError("PCM must have sample_rate=16000 and channels=1")
        if not isinstance(uid, str) or not uid or len(uid) > 256:
            raise HTTPException(422, "uid must contain 1 to 256 characters")
        if not isinstance(raw_candidates, str):
            raise HTTPException(422, "candidates must be a JSON string array")
        try:
            texts = json.loads(raw_candidates)
        except (ValueError, RecursionError) as error:
            raise HTTPException(422, "candidates must be a JSON string array") from error
        if not isinstance(texts, list) or not texts or any(not isinstance(t, str) for t in texts):
            raise HTTPException(422, "candidates must be a nonempty JSON string array")
        if len(texts) > settings.model.max_candidates:
            raise InputTooLarge("too many candidates")
        if any(len(text) > settings.model.max_text_chars for text in texts):
            raise InputTooLarge("candidate exceeds max_text_chars")
        return uid, await upload.read(), 16000, texts


def create_app(settings: Settings, *, engine_loader=PPLScorer):
    @asynccontextmanager
    async def lifespan(app):
        cpu_executor = ThreadPoolExecutor(max_workers=settings.server.decode_workers,
                                          thread_name_prefix="ppl-cpu")
        gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ppl-gpu")
        scheduler = None
        try:
            loop = asyncio.get_running_loop()
            engine = await loop.run_in_executor(gpu_executor, engine_loader, settings)
            scheduler = PPLBatchScheduler(engine=engine, executor=gpu_executor, settings=settings)
            await scheduler.start()
            app.state.engine = engine
            app.state.scheduler = scheduler
            app.state.cpu_executor = cpu_executor
            yield
        finally:
            if scheduler is not None:
                await scheduler.stop()
            await asyncio.to_thread(cpu_executor.shutdown, wait=True, cancel_futures=True)
            await asyncio.to_thread(gpu_executor.shutdown, wait=True, cancel_futures=True)

    app = FastAPI(title="FireRed PPL", lifespan=lifespan)

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        if not app.state.scheduler.is_healthy:
            raise HTTPException(503, "PPL worker is not ready; restart if inference failed")
        return {"status": "ok", "scorer_type": "aed", "audio_format": "pcm_s16le",
                **settings.runtime.model_dump()}

    @app.get("/stats")
    async def stats():
        return {"scheduler": app.state.scheduler.snapshot(),
                "scorer": dict(getattr(app.state.engine, "stats", {}))}

    @app.post("/score")
    async def score(request: Request):
        scheduler = app.state.scheduler
        reservation = preprocessing = result_future = None
        transferred = False
        try:
            async with asyncio.timeout(settings.server.request_timeout_s):
                reservation = scheduler.reserve()
                uid, pcm_bytes, sample_rate, texts = await parse_request(request, settings)
                scheduler.set_workload(reservation, len(texts), 0)
                preprocessing = asyncio.get_running_loop().run_in_executor(
                    app.state.cpu_executor, prepare_pcm, app.state.engine, settings,
                    uid, pcm_bytes, sample_rate, texts,
                )
                # Cancellation must not release capacity while a CPU worker is still running.
                prepared = await asyncio.shield(preprocessing)
                result_future = scheduler.submit(reservation, prepared)
                transferred = True
                results = await result_future
                return {"uid": uid, "scorer_type": "aed", "results": results}
        except QueueFullError as error:
            raise HTTPException(429, str(error), headers={"Retry-After": "1"}) from error
        except InputTooLarge as error:
            raise HTTPException(413, str(error)) from error
        except InputError as error:
            raise HTTPException(400, str(error)) from error
        except TimeoutError as error:
            raise HTTPException(504, "PPL request timed out") from error
        except (SchedulerClosedError, torch.cuda.OutOfMemoryError) as error:
            raise HTTPException(503, "PPL inference unavailable", headers={"Retry-After": "1"}) from error
        except StarletteHTTPException:
            raise
        except Exception as error:
            # Server tracebacks stay in logs, never in API response bodies.
            import logging
            logging.getLogger(__name__).exception("PPL request failed")
            raise HTTPException(500, "PPL request failed") from error
        finally:
            if result_future is not None and not result_future.done():
                result_future.cancel()
            if reservation is not None and not transferred:
                if preprocessing is not None:
                    def release_when_done(future):
                        if not future.cancelled():
                            future.exception()  # Consume errors from work whose caller timed out.
                        scheduler.release(reservation)
                    preprocessing.add_done_callback(release_when_done)
                else:
                    scheduler.release(reservation)

    return app


def parse_settings(argv=None):
    parser = argparse.ArgumentParser(description="FireRedASR2-AED PPL service")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-dir")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--encoder-precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--decoder-precision", choices=("fp32", "fp16", "bf16"))
    args = parser.parse_args(argv)
    data = yaml.safe_load(args.config.read_text()) if args.config else {}
    if not isinstance(data, dict):
        parser.error("config must be a YAML mapping")
    for section, fields in {"model": ("model_dir",), "server": ("host", "port"),
                            "runtime": ("use_gpu", "encoder_precision", "decoder_precision")}.items():
        for field in fields:
            value = getattr(args, field)
            if value is not None:
                section_data = data.setdefault(section, {})
                if not isinstance(section_data, dict):
                    parser.error(f"{section} must be a mapping")
                section_data[field] = value
    settings = Settings.model_validate(data)
    if not settings.model.model_dir:
        parser.error("provide --model-dir or model.model_dir in YAML")
    return settings


def main(argv=None):
    import uvicorn

    settings = parse_settings(argv)
    uvicorn.run(create_app(settings), host=settings.server.host, port=settings.server.port, workers=1)


if __name__ == "__main__":
    main()
