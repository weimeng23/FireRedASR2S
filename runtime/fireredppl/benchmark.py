"""Benchmark real audio/candidate distributions; uses only the Python standard library."""

import argparse
import io
import json
import math
import time
import urllib.error
import urllib.request
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def load_manifest(path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        wav = Path(item["wav_path"])
        if not wav.is_absolute():
            wav = path.parent / wav
        audio = wav.read_bytes()
        texts = item["texts"]
        if not isinstance(texts, list) or not texts or any(not isinstance(t, str) for t in texts):
            raise ValueError("texts must be a nonempty string array")
        with wave.open(io.BytesIO(audio)) as stream:
            if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getframerate() != 16000:
                raise ValueError("benchmark WAV sources must be mono 16 kHz PCM16")
            sample_rate = stream.getframerate()
            duration = stream.getnframes() / sample_rate
            pcm = stream.readframes(stream.getnframes())
            if not pcm or len(pcm) != stream.getnframes() * 2:
                raise ValueError("benchmark WAV source is empty or truncated")
        rows.append({"uid": str(item["uid"]), "pcm": pcm, "sample_rate": sample_rate,
                     "texts": texts, "duration_s": duration})
    if not rows:
        raise ValueError("manifest is empty")
    return rows


def multipart(item):
    boundary = uuid.uuid4().hex
    chunks = []
    for key, value in [("uid", item["uid"]), ("candidates", json.dumps(item["texts"], ensure_ascii=False)),
                       ("sample_rate", str(item["sample_rate"])), ("channels", "1"), ("sample_format", "s16le")]:
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        chunks.extend([value.encode(), b"\r\n"])
    chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="pcm"; filename="audio.pcm"\r\n'
                  'Content-Type: application/octet-stream\r\n\r\n'.encode())
    chunks.extend([item["pcm"], f"\r\n--{boundary}--\r\n".encode()])
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def call_score(url, item, timeout):
    body, content_type = multipart(item)
    request = urllib.request.Request(url, data=body, headers={"Content-Type": content_type})
    # Match the pipeline client's trust_env=False behavior.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        result = json.load(response)
    scores = result.get("results", [])
    if [score.get("text") for score in scores] != item["texts"]:
        raise ValueError("response candidate count/order mismatch")
    if any(not isinstance(score.get("ppl"), (int, float)) or not math.isfinite(score["ppl"])
           or score["ppl"] <= 0 for score in scores):
        raise ValueError("response contains invalid/null PPL")
    return result


def benchmark(url, rows, concurrency, repeats, timeout):
    def run(index):
        item = rows[index % len(rows)]
        start = time.perf_counter()
        try:
            call_score(url, item, timeout)
            error = None
        except Exception as exc:
            error = str(exc)
        return {"elapsed_s": time.perf_counter() - start, "error": error,
                "audio_s": item["duration_s"], "candidates": len(item["texts"])}

    start = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as executor:
        results = list(executor.map(run, range(len(rows) * repeats)))
    elapsed = time.perf_counter() - start
    good = [result for result in results if result["error"] is None]
    latencies = sorted(result["elapsed_s"] for result in good)

    def percentile(fraction):
        return 1000 * latencies[max(0, math.ceil(len(latencies) * fraction) - 1)] if latencies else None

    return {"concurrency": concurrency, "requests": len(results), "successful_requests": len(good),
            "failed_requests": len(results) - len(good), "elapsed_s": elapsed,
            "successful_requests_per_s": len(good) / elapsed,
            "audio_seconds_per_s": sum(r["audio_s"] for r in good) / elapsed,
            "candidates_per_s": sum(r["candidates"] for r in good) / elapsed,
            "p50_ms": percentile(0.50), "p95_ms": percentile(0.95), "p99_ms": percentile(0.99),
            "errors": [r["error"] for r in results if r["error"]][:10]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="full scoring endpoint URL, e.g. http://127.0.0.1:12345/score")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.concurrency) < 1 or args.repeats < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("concurrency/repeats/timeout must be positive; warmup must be nonnegative")
    rows = load_manifest(args.manifest)
    url = args.url
    for index in range(args.warmup):
        call_score(url, rows[index % len(rows)], args.timeout)
    reports = []
    for concurrency in args.concurrency:
        report = benchmark(url, rows, concurrency, args.repeats, args.timeout)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.output:
        args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    if any(report["failed_requests"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
