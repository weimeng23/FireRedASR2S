#!/usr/bin/env python3

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fireredasr2s.fireredlid.lid import load_fireredlid_model
from fireredasr2s.fireredlid.runtime.provenance import (
    collect_provenance,
    engine_input_artifacts,
    file_artifact,
    onnx_bundle_artifact,
)


def verify_onnx_outputs(
    encoder,
    session,
    features,
    lengths,
    rtol=1e-3,
    atol=1e-4,
):
    with torch.inference_mode():
        expected_outputs, expected_lengths, expected_mask = encoder(
            features,
            lengths,
        )
    actual_outputs, actual_lengths, actual_mask = session.run(
        None,
        {
            "features": features.cpu().numpy(),
            "feature_lengths": lengths.cpu().numpy(),
        },
    )
    actual_outputs = torch.from_numpy(actual_outputs)
    actual_lengths = torch.from_numpy(actual_lengths)
    actual_mask = torch.from_numpy(actual_mask)
    return _compare_outputs(
        expected_outputs.cpu(),
        expected_lengths.cpu(),
        expected_mask.cpu(),
        actual_outputs,
        actual_lengths,
        actual_mask,
        rtol=rtol,
        atol=atol,
    )


def _compare_outputs(
    expected_outputs,
    expected_lengths,
    expected_mask,
    actual_outputs,
    actual_lengths,
    actual_mask,
    rtol,
    atol,
):
    if not torch.equal(actual_mask, expected_mask):
        raise AssertionError("encoder_mask differs from eager baseline")
    if not torch.equal(actual_lengths, expected_lengths):
        raise AssertionError("encoder_lengths differs from eager baseline")
    torch.testing.assert_close(
        actual_outputs,
        expected_outputs,
        rtol=rtol,
        atol=atol,
    )
    max_abs_error = float(
        (actual_outputs - expected_outputs).abs().max().item()
    )
    return {"max_abs_error": max_abs_error}


def verify_backend_outputs(
    encoder,
    backend,
    features,
    lengths,
    rtol=2e-2,
    atol=2e-2,
):
    with torch.inference_mode():
        expected_outputs, expected_lengths, expected_mask = encoder(
            features,
            lengths,
        )
        actual = backend.encode(features, lengths)
    return _compare_outputs(
        expected_outputs,
        expected_lengths,
        expected_mask,
        actual.outputs,
        actual.lengths,
        actual.mask,
        rtol=rtol,
        atol=atol,
    )


def resolve_tolerances(args):
    default_rtol, default_atol = (
        (1e-3, 1e-4) if args.backend == "onnx" else (2e-2, 2e-2)
    )
    rtol = default_rtol if args.rtol is None else args.rtol
    atol = default_atol if args.atol is None else args.atol
    return rtol, atol


def environment():
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
        description="Compare a FireRedLID Encoder backend against eager."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--backend",
        choices=["onnx", "tensorrt"],
        default="onnx",
    )
    artifacts = parser.add_mutually_exclusive_group(required=True)
    artifacts.add_argument("--onnx")
    artifacts.add_argument("--engine-dir")
    parser.add_argument("--report")
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument(
        "--seconds",
        type=int,
        nargs="+",
        default=[1, 5, 15, 30, 60],
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2],
    )
    args = parser.parse_args(argv)
    if args.backend == "onnx" and not args.onnx:
        parser.error("--backend onnx requires --onnx")
    if args.backend == "tensorrt" and not args.engine_dir:
        parser.error("--backend tensorrt requires --engine-dir")
    for name in ("rtol", "atol"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name} must be finite and non-negative")
    return args


def main():
    args = parse_args()
    rtol, atol = resolve_tolerances(args)
    checkpoint_path = Path(args.model_dir) / "model.pth.tar"
    model = load_fireredlid_model(checkpoint_path)
    if args.backend == "onnx":
        import onnxruntime as ort

        encoder = model.encoder.float().cpu().eval()
        session = ort.InferenceSession(
            args.onnx,
            providers=["CPUExecutionProvider"],
        )
        backend = None
    else:
        from fireredasr2s.fireredlid.runtime.tensorrt_backend import (
            TensorRTEncoderBackend,
        )

        encoder = model.encoder.half().cuda().eval()
        session = None
        backend = TensorRTEncoderBackend(
            args.engine_dir,
            checkpoint_path=checkpoint_path,
        )
    torch.manual_seed(0)
    cases = []
    failed = False
    for seconds in args.seconds:
        frames = seconds * 100
        for batch_size in args.batch_sizes:
            features = torch.randn(batch_size, frames, 80)
            lengths = torch.full(
                (batch_size,),
                frames,
                dtype=torch.long,
            )
            if batch_size > 1:
                lengths[-1] = max(1, frames - 37)
            if args.backend == "tensorrt":
                features = features.half().cuda()
                lengths = lengths.cuda()
            started = time.perf_counter()
            try:
                if args.backend == "onnx":
                    metrics = verify_onnx_outputs(
                        encoder,
                        session,
                        features,
                        lengths,
                        rtol=rtol,
                        atol=atol,
                    )
                else:
                    metrics = verify_backend_outputs(
                        encoder,
                        backend,
                        features,
                        lengths,
                        rtol=rtol,
                        atol=atol,
                    )
                cases.append(
                    {
                        "seconds": seconds,
                        "batch_size": batch_size,
                        "frames": frames,
                        "elapsed_s": time.perf_counter() - started,
                        "status": "passed",
                        **metrics,
                    }
                )
            except Exception as error:
                failed = True
                cases.append(
                    {
                        "seconds": seconds,
                        "batch_size": batch_size,
                        "frames": frames,
                        "elapsed_s": time.perf_counter() - started,
                        "status": "failed",
                        "error": str(error),
                    }
                )
    report_path = (
        Path(args.report)
        if args.report
        else (
            Path(args.onnx).with_name("verify.fp32.json")
            if args.backend == "onnx"
            else Path(args.engine_dir) / "verify.fp16.json"
        )
    )
    report_arguments = {
        **vars(args),
        "resolved_rtol": rtol,
        "resolved_atol": atol,
    }
    input_artifacts = {"checkpoint": file_artifact(checkpoint_path)}
    if args.backend == "onnx":
        input_artifacts["onnx_bundle"] = onnx_bundle_artifact(args.onnx)
    else:
        input_artifacts.update(engine_input_artifacts(args.engine_dir))
    report = {
        "arguments": report_arguments,
        "environment": environment(),
        "provenance": collect_provenance(
            report_arguments,
            input_artifacts,
            REPO_ROOT,
        ),
        "cases": cases,
        "passed": not failed,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
