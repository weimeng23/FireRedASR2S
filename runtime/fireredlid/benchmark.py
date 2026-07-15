#!/usr/bin/env python3

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fireredasr2s.fireredlid.runtime.provenance import (
    collect_provenance,
    engine_input_artifacts,
    file_artifact,
    model_input_artifacts,
)
from fireredasr2s.fireredlid.runtime.benchmark_config import (
    resolve_device,
    validate_backend_device_precision,
)


class StageRecorder:
    def __init__(self, synchronize_cuda):
        self.synchronize_cuda = synchronize_cuda
        self.total_seconds = defaultdict(float)
        self.calls = defaultdict(int)

    def _synchronize(self):
        if self.synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def measure(self, name):
        self._synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            self.total_seconds[name] += time.perf_counter() - started
            self.calls[name] += 1

    def reset(self):
        self.total_seconds.clear()
        self.calls.clear()


def summarize(latencies_s, utterances, audio_seconds, elapsed_s):
    values_ms = np.asarray(latencies_s, dtype=np.float64) * 1000.0
    return {
        "p50_ms": round(float(np.percentile(values_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(values_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(values_ms, 99)), 3),
        "utterances_per_s": round(utterances / elapsed_s, 3),
        "audio_seconds_per_s": round(audio_seconds / elapsed_s, 3),
        "rtf": round(elapsed_s / audio_seconds, 6),
    }


def load_manifest(path):
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("uttid"), str) or not isinstance(
                record.get("wav"), str
            ):
                raise ValueError(
                    f"manifest line {line_number} requires string uttid and wav"
                )
            records.append(
                {"uttid": record["uttid"], "wav": record["wav"]}
            )
    if not records:
        raise ValueError("manifest must not be empty")
    return records


def _chunks(records, size):
    return [
        records[start : start + size]
        for start in range(0, len(records), size)
    ]


def _synchronize_cuda(enabled):
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _planner(lid):
    from fireredasr2s.fireredlid.runtime.batch_planner import BatchPlanner

    limits = [
        value
        for value in (
            lid.config.max_sub_batch_size,
            lid.backend_max_batch,
        )
        if value is not None
    ]
    return BatchPlanner(
        strategy=lid.config.resolved_batch_strategy,
        max_sub_batch_size=min(limits) if limits else None,
    )


def _plan_report(lid, prepared_groups):
    shapes = []
    valid_frames = 0
    padded_frames = 0
    physical_batches = 0
    planner = _planner(lid)
    for items in prepared_groups:
        for planned in planner.plan(items):
            batch, frames, feature_dim = planned.padded_features.shape
            shapes.append([batch, frames, feature_dim])
            physical_batches += 1
            valid_frames += int(planned.feature_lengths.sum().item())
            padded_frames += batch * frames
    padding_ratio = (
        (padded_frames - valid_frames) / padded_frames
        if padded_frames
        else 0.0
    )
    return {
        "logical_batch_count": len(prepared_groups),
        "physical_batch_count": physical_batches,
        "actual_input_shapes": shapes,
        "valid_frames": valid_frames,
        "padded_frames": padded_frames,
        "padding_ratio": round(padding_ratio, 6),
    }


def _run_encoder(lid, items):
    outputs = []
    for planned in _planner(lid).plan(items):
        with lid._measure_stage("h2d"):
            features = planned.padded_features
            lengths = planned.feature_lengths
            if lid.config.use_gpu:
                features = features.cuda()
                lengths = lengths.cuda()
                if lid.config.use_half:
                    features = features.half()
        with lid._measure_stage("encoder"):
            outputs.append(lid.model.encoder(features, lengths))
    return outputs


def _run_workload(
    lid,
    record_groups,
    prepared_groups,
    scope,
    synchronize_cuda,
):
    latencies = []
    utterances = 0
    audio_seconds = 0.0
    for records, items in zip(record_groups, prepared_groups):
        _synchronize_cuda(synchronize_cuda)
        started = time.perf_counter()
        if scope == "end-to-end":
            result = lid.process(
                [record["uttid"] for record in records],
                [record["wav"] for record in records],
            )
            if len(result) != len(records):
                raise RuntimeError(
                    "end-to-end inference returned an unexpected result count"
                )
        elif scope == "model":
            result = lid._infer_items(items)
            if len(result) != len(items):
                raise RuntimeError(
                    "model inference returned an unexpected result count"
                )
        else:
            _run_encoder(lid, items)
        _synchronize_cuda(synchronize_cuda)
        latencies.append(time.perf_counter() - started)
        utterances += len(items)
        audio_seconds += sum(item.duration_s for item in items)
    return latencies, utterances, audio_seconds


def _environment():
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": gpu,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark FireRedLID eager, compile, or TensorRT inference."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--backend",
        choices=["eager", "compile", "tensorrt"],
        default="eager",
    )
    parser.add_argument(
        "--profile",
        choices=["latency", "throughput"],
        default="latency",
    )
    parser.add_argument(
        "--scope",
        choices=["encoder", "model", "end-to-end"],
        default="end-to-end",
    )
    parser.add_argument(
        "--batch-strategy",
        choices=["none", "bucket", "auto"],
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp32",
    )
    parser.add_argument("--engine-dir")
    parser.add_argument("--max-sub-batch-size", type=int)
    parser.add_argument("--logical-batch-size", type=int)
    parser.add_argument("--max-audio-seconds", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        validate_backend_device_precision(
            [args.backend],
            args.device,
            args.precision,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.backend == "tensorrt" and not args.engine_dir:
        parser.error("TensorRT benchmark requires --engine-dir")
    return args


def main():
    args = parse_args()
    validate_backend_device_precision(
        [args.backend],
        args.device,
        args.precision,
    )
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations positive")
    if args.backend == "tensorrt" and not args.engine_dir:
        raise ValueError("TensorRT benchmark requires --engine-dir")

    resolved_device = resolve_device(
        args.device,
        torch.cuda.is_available(),
    )
    resolved_precision = args.precision
    use_gpu = resolved_device == "cuda"
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if resolved_precision == "fp16" and not use_gpu:
        raise ValueError("FP16 benchmark requires CUDA")

    records = load_manifest(args.manifest)
    logical_batch_size = args.logical_batch_size
    if logical_batch_size is None:
        logical_batch_size = 1 if args.profile == "latency" else len(records)
    if logical_batch_size < 1:
        raise ValueError("logical_batch_size must be positive")
    record_groups = _chunks(records, logical_batch_size)

    from fireredasr2s.fireredlid.lid import FireRedLid, FireRedLidConfig

    config = FireRedLidConfig(
        use_gpu=use_gpu,
        use_half=resolved_precision == "fp16",
        backend=args.backend,
        profile=args.profile,
        max_audio_seconds=args.max_audio_seconds,
        batch_strategy=args.batch_strategy,
        engine_dir=args.engine_dir,
        max_sub_batch_size=args.max_sub_batch_size,
    )
    lid = FireRedLid.from_pretrained(args.model_dir, config)
    prepared_groups = []
    for group in record_groups:
        items = lid.feat_extractor.extract_many(
            [record["wav"] for record in group],
            [record["uttid"] for record in group],
            max_audio_seconds=config.max_audio_seconds,
        )
        if len(items) != len(group):
            raise RuntimeError(
                "feature extraction returned an unexpected item count"
            )
        prepared_groups.append(items)

    synchronize_cuda = use_gpu
    recorder = StageRecorder(synchronize_cuda=synchronize_cuda)
    lid.stage_recorder = recorder
    plan_report = _plan_report(lid, prepared_groups)

    first_started = time.perf_counter()
    _run_workload(
        lid,
        record_groups,
        prepared_groups,
        args.scope,
        synchronize_cuda,
    )
    first_run_s = time.perf_counter() - first_started

    warmup_started = time.perf_counter()
    for _ in range(args.warmup):
        _run_workload(
            lid,
            record_groups,
            prepared_groups,
            args.scope,
            synchronize_cuda,
        )
    warmup_s = time.perf_counter() - warmup_started

    recorder.reset()
    if use_gpu:
        torch.cuda.reset_peak_memory_stats()
    stable_latencies = []
    total_utterances = 0
    total_audio_seconds = 0.0
    stable_started = time.perf_counter()
    for _ in range(args.iterations):
        latencies, utterances, audio_seconds = _run_workload(
            lid,
            record_groups,
            prepared_groups,
            args.scope,
            synchronize_cuda,
        )
        stable_latencies.extend(latencies)
        total_utterances += utterances
        total_audio_seconds += audio_seconds
    stable_elapsed_s = time.perf_counter() - stable_started
    peak_gpu_bytes = (
        int(torch.cuda.max_memory_allocated()) if use_gpu else None
    )

    report_arguments = {
        **vars(args),
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "requested_precision": args.precision,
        "resolved_precision": resolved_precision,
    }
    input_artifacts = model_input_artifacts(args.model_dir)
    input_artifacts["audio_manifest"] = file_artifact(args.manifest)
    if args.backend == "tensorrt":
        input_artifacts.update(engine_input_artifacts(args.engine_dir))
    report = {
        "arguments": report_arguments,
        "environment": _environment(),
        "provenance": collect_provenance(
            report_arguments,
            input_artifacts,
            REPO_ROOT,
        ),
        "requested_backend": args.backend,
        "active_backend": lid.active_backend,
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "requested_precision": args.precision,
        "resolved_precision": resolved_precision,
        "first_run_s": first_run_s,
        "warmup_s": warmup_s,
        "stable_elapsed_s": stable_elapsed_s,
        "stable_iterations": args.iterations,
        "peak_gpu_bytes": peak_gpu_bytes,
        "stages": {
            name: {
                "total_s": recorder.total_seconds[name],
                "calls": recorder.calls[name],
                "mean_ms": (
                    recorder.total_seconds[name]
                    / recorder.calls[name]
                    * 1000.0
                ),
            }
            for name in sorted(recorder.total_seconds)
        },
        **plan_report,
        **summarize(
            stable_latencies,
            total_utterances,
            total_audio_seconds,
            stable_elapsed_s,
        ),
    }
    output = (
        Path(args.output)
        if args.output
        else Path("runtime/fireredlid/artifacts")
        / f"benchmark.{args.backend}.{args.profile}.{args.scope}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
