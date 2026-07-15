#!/usr/bin/env python3

import argparse
import json
import platform
import shlex
import subprocess
import sys
from pathlib import Path


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


BENCHMARK_SCRIPT = Path(__file__).resolve().with_name("benchmark.py")
BACKENDS = ["eager", "compile", "tensorrt"]
SCOPES = ["encoder", "model", "end-to-end"]
PROFILE_WORKLOADS = {
    "latency": {
        "batch_strategy": "none",
        "logical_batch_size": 1,
        "warmup": 5,
        "iterations": 50,
    },
    "throughput": {
        "batch_strategy": "auto",
        "logical_batch_size": 100,
        "warmup": 3,
        "iterations": 20,
    },
}


def _validate_unique_selectors(args):
    for name in ("backends", "scopes"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            raise ValueError(f"--{name} must not contain duplicates")


def build_commands(args) -> list[list[str]]:
    _validate_unique_selectors(args)
    validate_backend_device_precision(
        args.backends,
        args.device,
        args.precision,
    )
    if "tensorrt" in args.backends and not args.engine_dir:
        raise ValueError("TensorRT matrix requires --engine-dir")

    workload = PROFILE_WORKLOADS[args.profile]
    commands = []
    for backend in args.backends:
        for scope in args.scopes:
            output = (
                Path(args.output_dir)
                / f"benchmark.{backend}.{args.profile}.{scope}.json"
            )
            command = [
                sys.executable,
                str(BENCHMARK_SCRIPT),
                "--model-dir",
                str(args.model_dir),
                "--manifest",
                str(args.manifest),
                "--backend",
                backend,
                "--device",
                args.device,
                "--precision",
                args.precision,
                "--profile",
                args.profile,
                "--batch-strategy",
                workload["batch_strategy"],
                "--logical-batch-size",
                str(workload["logical_batch_size"]),
                "--scope",
                scope,
                "--warmup",
                str(workload["warmup"]),
                "--iterations",
                str(workload["iterations"]),
                "--output",
                str(output),
            ]
            if backend == "tensorrt":
                command.extend(["--engine-dir", str(args.engine_dir)])
            commands.append(command)
    return commands


def _environment():
    import torch

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


def _validate_execute_paths(args):
    model_dir = Path(args.model_dir)
    manifest = Path(args.manifest)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest file not found: {manifest}")
    if "tensorrt" in args.backends:
        engine_dir = Path(args.engine_dir)
        if not engine_dir.is_dir():
            raise FileNotFoundError(
                f"engine directory not found: {engine_dir}"
            )


def _option_value(command, option):
    return command[command.index(option) + 1]


def _write_index(path, index):
    path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate or execute a deterministic FireRedLID Linux "
            "benchmark matrix."
        )
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_WORKLOADS),
        default="latency",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKENDS,
        default=BACKENDS,
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=SCOPES,
        default=SCOPES,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp16",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        _validate_unique_selectors(args)
        validate_backend_device_precision(
            args.backends,
            args.device,
            args.precision,
        )
    except ValueError as error:
        parser.error(str(error))
    if "tensorrt" in args.backends and not args.engine_dir:
        parser.error("TensorRT in --backends requires --engine-dir")
    return args


def main(argv=None):
    args = parse_args(argv)
    commands = build_commands(args)
    if args.execute:
        _validate_execute_paths(args)

    for command in commands:
        print(shlex.join(command))
    if not args.execute:
        return 0

    import torch

    resolved_device = resolve_device(
        args.device,
        torch.cuda.is_available(),
    )
    resolved_precision = args.precision
    report_arguments = {
        "model_dir": str(args.model_dir),
        "manifest": str(args.manifest),
        "engine_dir": (
            str(args.engine_dir) if args.engine_dir is not None else None
        ),
        "profile": args.profile,
        "backends": list(args.backends),
        "scopes": list(args.scopes),
        "device": args.device,
        "precision": args.precision,
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "requested_precision": args.precision,
        "resolved_precision": resolved_precision,
        "output_dir": str(args.output_dir),
        "execute": args.execute,
    }
    input_artifacts = model_input_artifacts(args.model_dir)
    input_artifacts["audio_manifest"] = file_artifact(args.manifest)
    if "tensorrt" in args.backends:
        input_artifacts.update(engine_input_artifacts(args.engine_dir))
    provenance = collect_provenance(
        report_arguments,
        input_artifacts,
        REPO_ROOT,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "matrix.index.json"
    index = {
        "schema_version": 1,
        "arguments": report_arguments,
        "profile": args.profile,
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "requested_precision": args.precision,
        "resolved_precision": resolved_precision,
        "environment": _environment(),
        "commit_hash": provenance["git"]["commit"] or "unknown",
        "provenance": provenance,
        "runs": [],
    }
    for command in commands:
        completed = subprocess.run(command, check=False)
        index["runs"].append(
            {
                "command": command,
                "output_path": _option_value(command, "--output"),
                "return_code": completed.returncode,
            }
        )
        _write_index(index_path, index)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
