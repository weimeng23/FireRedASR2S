#!/usr/bin/env python3

import argparse
import gc
import json
import math
import platform
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

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


def _decimal_confidence(value, source, uttid):
    try:
        confidence = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(
            f"{source} result {uttid!r} has invalid confidence"
        ) from None
    if not confidence.is_finite():
        raise ValueError(
            f"{source} result {uttid!r} has non-finite confidence"
        )
    return confidence


def _index_results(results, source):
    counts = Counter()
    validated = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"{source} result {index} must be a dictionary")
        uttid = result.get("uttid")
        if not isinstance(uttid, str):
            raise ValueError(f"{source} result {index} requires string uttid")
        if not isinstance(result.get("lang"), str):
            raise ValueError(f"{source} result {uttid!r} requires string lang")
        confidence = _decimal_confidence(
            result.get("confidence"),
            source,
            uttid,
        )
        counts[uttid] += 1
        validated.append((uttid, result, confidence))

    duplicates = sorted(
        uttid for uttid, count in counts.items() if count > 1
    )
    duplicate_set = set(duplicates)
    indexed = {
        uttid: (result, confidence)
        for uttid, result, confidence in validated
        if uttid not in duplicate_set
    }
    return indexed, duplicates, set(counts)


def compare_results(
    baseline: list[dict],
    candidate: list[dict],
    confidence_atol: float,
) -> dict:
    try:
        tolerance = Decimal(str(confidence_atol))
    except (InvalidOperation, ValueError):
        raise ValueError(
            "confidence_atol must be a finite non-negative number"
        ) from None
    if not tolerance.is_finite() or tolerance < 0:
        raise ValueError(
            "confidence_atol must be a finite non-negative number"
        )

    baseline_by_id, baseline_duplicates, baseline_ids = _index_results(
        baseline,
        "baseline",
    )
    candidate_by_id, candidate_duplicates, candidate_ids = _index_results(
        candidate,
        "candidate",
    )
    incomparable_uttids = set(baseline_duplicates + candidate_duplicates)
    comparable_uttids = sorted(
        (baseline_ids & candidate_ids) - incomparable_uttids
    )
    missing_in_baseline = sorted(candidate_ids - baseline_ids)
    missing_in_candidate = sorted(baseline_ids - candidate_ids)

    label_mismatches = []
    confidence_mismatches = []
    confidence_errors = []
    for uttid in comparable_uttids:
        baseline_result, baseline_confidence = baseline_by_id[uttid]
        candidate_result, candidate_confidence = candidate_by_id[uttid]
        if baseline_result["lang"] != candidate_result["lang"]:
            label_mismatches.append(
                {
                    "uttid": uttid,
                    "baseline_lang": baseline_result["lang"],
                    "candidate_lang": candidate_result["lang"],
                }
            )
        confidence_error = abs(
            baseline_confidence - candidate_confidence
        )
        confidence_errors.append(confidence_error)
        if confidence_error > tolerance:
            confidence_mismatches.append(
                {
                    "uttid": uttid,
                    "baseline_confidence": float(baseline_confidence),
                    "candidate_confidence": float(candidate_confidence),
                    "abs_error": float(confidence_error),
                }
            )

    passed = not any(
        (
            baseline_duplicates,
            candidate_duplicates,
            missing_in_baseline,
            missing_in_candidate,
            label_mismatches,
            confidence_mismatches,
        )
    )
    return {
        "passed": passed,
        "confidence_atol": float(tolerance),
        "counts": {
            "baseline": len(baseline),
            "candidate": len(candidate),
            "compared": len(comparable_uttids),
        },
        "max_confidence_abs_error": (
            float(max(confidence_errors)) if confidence_errors else 0.0
        ),
        "duplicate_uttids": {
            "baseline": baseline_duplicates,
            "candidate": candidate_duplicates,
        },
        "missing_in_baseline": missing_in_baseline,
        "missing_in_candidate": missing_in_candidate,
        "label_mismatches": label_mismatches,
        "confidence_mismatches": confidence_mismatches,
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
                    f"manifest line {line_number} requires string "
                    "uttid and wav"
                )
            records.append(
                {"uttid": record["uttid"], "wav": record["wav"]}
            )
    if not records:
        raise ValueError("manifest must not be empty")
    return records


def _lightweight_results(results, expected_count):
    if len(results) != expected_count:
        raise RuntimeError("backend returned an unexpected result count")
    lightweight = []
    for result in results:
        try:
            lightweight.append(
                {
                    "uttid": result["uttid"],
                    "lang": result["lang"],
                    "confidence": result["confidence"],
                }
            )
        except KeyError as error:
            raise RuntimeError(
                f"backend result is missing {error.args[0]!r}"
            ) from error
    return lightweight


def run_backend(records, model_dir, backend, engine_dir=None):
    from fireredasr2s.fireredlid.lid import FireRedLid, FireRedLidConfig

    config = FireRedLidConfig(
        use_gpu=True,
        backend=backend,
        profile="latency",
        encoder_precision="fp16",
        decoder_precision="fp16",
        engine_dir=engine_dir,
        fallback_backend=None,
    )
    model = FireRedLid.from_pretrained(model_dir, config)
    try:
        if model.active_backend != backend:
            raise RuntimeError(
                f"requested backend {backend!r} initialized as "
                f"{model.active_backend!r}"
            )
        results = []
        for record in records:
            raw_results = model.process(
                [record["uttid"]],
                [record["wav"]],
            )
            results.extend(_lightweight_results(raw_results, 1))
        return results
    finally:
        del model


def _environment():
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare FireRedLID compile or TensorRT final labels and "
            "confidence against eager FP16."
        )
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--candidate-backend",
        choices=["compile", "tensorrt"],
        required=True,
    )
    parser.add_argument("--engine-dir")
    parser.add_argument("--confidence-atol", type=float, default=0.005)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    if args.candidate_backend == "tensorrt" and not args.engine_dir:
        parser.error("--candidate-backend tensorrt requires --engine-dir")
    if not math.isfinite(args.confidence_atol) or args.confidence_atol < 0:
        parser.error("--confidence-atol must be finite and non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "final-label verification requires CUDA and must run on "
            "Linux NVIDIA"
        )

    records = load_manifest(args.manifest)
    try:
        baseline = run_backend(
            records,
            model_dir=args.model_dir,
            backend="eager",
            engine_dir=None,
        )
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    candidate_engine_dir = (
        args.engine_dir if args.candidate_backend == "tensorrt" else None
    )
    try:
        candidate = run_backend(
            records,
            model_dir=args.model_dir,
            backend=args.candidate_backend,
            engine_dir=candidate_engine_dir,
        )
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    report_arguments = {
        "model_dir": args.model_dir,
        "manifest": args.manifest,
        "baseline_backend": "eager",
        "candidate_backend": args.candidate_backend,
        "device": "cuda",
        "precision": "fp16",
        "engine_dir": candidate_engine_dir,
        "confidence_atol": args.confidence_atol,
        "report": args.report,
    }
    input_artifacts = model_input_artifacts(args.model_dir)
    input_artifacts["audio_manifest"] = file_artifact(args.manifest)
    if args.candidate_backend == "tensorrt":
        input_artifacts.update(engine_input_artifacts(args.engine_dir))
    provenance = collect_provenance(
        report_arguments,
        input_artifacts,
        REPO_ROOT,
    )
    report = {
        "schema_version": 1,
        "arguments": report_arguments,
        "environment": _environment(),
        "commit_hash": provenance["git"]["commit"] or "unknown",
        "provenance": provenance,
        **compare_results(
            baseline,
            candidate,
            confidence_atol=args.confidence_atol,
        ),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report_path)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
