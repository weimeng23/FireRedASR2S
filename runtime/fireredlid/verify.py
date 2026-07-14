#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fireredasr2s.fireredlid.lid import load_fireredlid_model


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
    rtol=1e-2,
    atol=1e-2,
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


def parse_args():
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
    args = parser.parse_args()
    if args.backend == "onnx" and not args.onnx:
        parser.error("--backend onnx requires --onnx")
    if args.backend == "tensorrt" and not args.engine_dir:
        parser.error("--backend tensorrt requires --engine-dir")
    return args


def main():
    args = parse_args()
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
                    )
                else:
                    metrics = verify_backend_outputs(
                        encoder,
                        backend,
                        features,
                        lengths,
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
    report_path.write_text(
        json.dumps({"cases": cases}, indent=2),
        encoding="utf-8",
    )
    print(report_path)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
