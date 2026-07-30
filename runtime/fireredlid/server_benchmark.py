#!/usr/bin/env python3

import argparse
import base64
import json
import math
import statistics
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen


SERVER_FIELDS = (
    "backend",
    "dtype",
    "encoder_dtype",
    "decoder_dtype",
)


def _wav_duration_s(audio_bytes, path):
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as source:
            frame_rate = source.getframerate()
            if frame_rate <= 0:
                raise ValueError("WAV frame rate must be positive")
            return source.getnframes() / frame_rate
    except (EOFError, wave.Error) as error:
        raise ValueError(f"failed to read WAV duration: {path}") from error


def load_manifest(path):
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            uttid = record.get("uttid")
            wav_path = record.get("wav")
            if not isinstance(uttid, str) or not isinstance(wav_path, str):
                raise ValueError(
                    f"manifest line {line_number} requires string uttid and wav"
                )
            audio_bytes = Path(wav_path).read_bytes()
            records.append(
                {
                    "uttid": uttid,
                    "wav": wav_path,
                    "audio_base64": base64.b64encode(audio_bytes).decode(
                        "ascii"
                    ),
                    "duration_s": _wav_duration_s(audio_bytes, wav_path),
                }
            )
    if not records:
        raise ValueError("manifest must not be empty")
    return records


def send_request(
    url,
    payload,
    timeout_s,
    *,
    open_url=urlopen,
    clock=time.perf_counter,
):
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = clock()
    with open_url(request, timeout=timeout_s) as response:
        result = json.loads(response.read())
    return result, clock() - started


def _build_request(records, request_index, request_batch_size):
    inputs = []
    audio_seconds = 0.0
    for item_index in range(request_batch_size):
        record_index = (
            request_index * request_batch_size + item_index
        ) % len(records)
        record = records[record_index]
        inputs.append(
            {
                "uttid": (
                    f"{record['uttid']}-request-{request_index}"
                    f"-item-{item_index}"
                ),
                "audio_base64": record["audio_base64"],
            }
        )
        audio_seconds += record["duration_s"]
    return (
        json.dumps({"inputs": inputs}).encode("utf-8"),
        [item["uttid"] for item in inputs],
        audio_seconds,
    )


def _validate_response(response, expected_uttids):
    if not isinstance(response, dict):
        raise RuntimeError("server response must be a JSON object")
    errors = response.get("errors", [])
    if errors:
        raise RuntimeError(
            f"server returned {len(errors)} item inference errors"
        )
    results = response.get("results")
    if not isinstance(results, list):
        raise RuntimeError("server response is missing results")
    returned_uttids = [
        result.get("uttid") if isinstance(result, dict) else None
        for result in results
    ]
    if returned_uttids != expected_uttids:
        raise RuntimeError("server returned unexpected result uttids")
    return {
        field: response.get(field)
        for field in SERVER_FIELDS
    }


def _execute_requests(
    records,
    *,
    url,
    concurrency,
    total_requests,
    request_batch_size,
    timeout_s,
    request_sender,
):
    if total_requests == 0:
        return [], []

    start_gate = threading.Event()

    def invoke(request_index):
        payload, expected_uttids, audio_seconds = _build_request(
            records,
            request_index,
            request_batch_size,
        )
        start_gate.wait()
        response, latency_s = request_sender(url, payload, timeout_s)
        server = _validate_response(response, expected_uttids)
        return {
            "request_index": request_index,
            "latency_s": latency_s,
            "items": request_batch_size,
            "audio_seconds": audio_seconds,
            "server": server,
        }

    outcomes = []
    errors = []
    with ThreadPoolExecutor(
        max_workers=min(concurrency, total_requests)
    ) as executor:
        futures = {
            executor.submit(invoke, request_index): request_index
            for request_index in range(total_requests)
        }
        start_gate.set()
        for future in as_completed(futures):
            request_index = futures[future]
            try:
                outcomes.append(future.result())
            except Exception as error:
                errors.append(
                    {
                        "request_index": request_index,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
    outcomes.sort(key=lambda item: item["request_index"])
    errors.sort(key=lambda item: item["request_index"])
    return outcomes, errors


def _percentile(values, quantile):
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _latency_summary(latencies_s):
    if not latencies_s:
        return None
    values = sorted(value * 1000.0 for value in latencies_s)
    return {
        "min": round(values[0], 3),
        "mean": round(statistics.fmean(values), 3),
        "p50": round(_percentile(values, 0.50), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "p99": round(_percentile(values, 0.99), 3),
        "max": round(values[-1], 3),
    }


def run_benchmark(
    records,
    *,
    url,
    concurrency,
    total_requests,
    request_batch_size,
    warmup_requests,
    timeout_s,
    request_sender=send_request,
    clock=time.perf_counter,
):
    warmup_outcomes, warmup_errors = _execute_requests(
        records,
        url=url,
        concurrency=concurrency,
        total_requests=warmup_requests,
        request_batch_size=request_batch_size,
        timeout_s=timeout_s,
        request_sender=request_sender,
    )
    if warmup_errors:
        error = warmup_errors[0]
        raise RuntimeError(
            "warmup request failed: "
            f"{error['error_type']}: {error['message']}"
        )
    if len(warmup_outcomes) != warmup_requests:
        raise RuntimeError("warmup did not complete every request")

    started = clock()
    outcomes, errors = _execute_requests(
        records,
        url=url,
        concurrency=concurrency,
        total_requests=total_requests,
        request_batch_size=request_batch_size,
        timeout_s=timeout_s,
        request_sender=request_sender,
    )
    elapsed_s = clock() - started
    if elapsed_s <= 0:
        raise RuntimeError("benchmark elapsed time must be positive")

    successful_requests = len(outcomes)
    failed_requests = len(errors)
    successful_items = sum(item["items"] for item in outcomes)
    total_items = total_requests * request_batch_size
    audio_seconds = sum(item["audio_seconds"] for item in outcomes)
    server = (
        outcomes[0]["server"]
        if outcomes
        else {field: None for field in SERVER_FIELDS}
    )
    summary = {
        "elapsed_s": round(elapsed_s, 6),
        "requests_total": total_requests,
        "requests_succeeded": successful_requests,
        "requests_failed": failed_requests,
        "request_error_rate": round(
            failed_requests / total_requests,
            6,
        ),
        "requests_per_s": round(successful_requests / elapsed_s, 3),
        "items_total": total_items,
        "items_succeeded": successful_items,
        "items_failed": total_items - successful_items,
        "items_per_s": round(successful_items / elapsed_s, 3),
        "audio_seconds_succeeded": round(audio_seconds, 6),
        "audio_seconds_per_s": round(audio_seconds / elapsed_s, 3),
        "request_latency_ms": _latency_summary(
            [item["latency_s"] for item in outcomes]
        ),
    }
    return {
        "schema_version": 1,
        "arguments": {
            "url": url,
            "concurrency": concurrency,
            "requests": total_requests,
            "request_batch_size": request_batch_size,
            "warmup_requests": warmup_requests,
            "timeout_s": timeout_s,
        },
        "server": server,
        "summary": summary,
        "errors": errors,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Benchmark the FireRedLID FastAPI endpoint",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:12345/v1/lid",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--request-batch-size", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.requests <= 0:
        parser.error("--requests must be positive")
    if args.request_batch_size <= 0:
        parser.error("--request-batch-size must be positive")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be non-negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    records = load_manifest(args.manifest)
    report = run_benchmark(
        records,
        url=args.url,
        concurrency=args.concurrency,
        total_requests=args.requests,
        request_batch_size=args.request_batch_size,
        warmup_requests=args.warmup_requests,
        timeout_s=args.timeout,
    )
    report["manifest"] = {
        "path": str(Path(args.manifest)),
        "records": len(records),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    return 1 if report["summary"]["requests_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
