"""Compare the batched service scorer with an explicitly supplied legacy scorer."""

import argparse
import asyncio
import gc
import importlib.util
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
import torch

# Allows invocation as a script from an uninstalled source checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fireredasr2s.fireredppl.config import Settings
from fireredasr2s.fireredppl.scheduler import PPLBatchScheduler
from fireredasr2s.fireredppl.scorer import PPLScorer
from fireredasr2s.fireredppl.server import prepare_pcm


def score_fields(result):
    """Keep numeric score data, excluding the legacy waveform-valued wav_path."""
    fields = {key: result[key] for key in ("uid", "text", "ppl", "avg_nll", "total_nll", "token_count")}
    return {key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in fields.items()}


def compare(reference, actual, *, atol, rtol):
    failures, ranking_changes, max_error = [], [], 0.0
    if len(reference) != len(actual):
        raise ValueError("audio result count mismatch")
    for index, (old, new) in enumerate(zip(reference, actual, strict=True)):
        if len(old) != len(new):
            raise ValueError("candidate result count mismatch")
        valid = True
        for candidate, (a, b) in enumerate(zip(old, new, strict=True)):
            if a["text"] != b["text"] or a["token_count"] != b["token_count"]:
                failures.append({"audio": index, "candidate": candidate, "reason": "text/token mismatch"})
            x, y = a.get("avg_nll"), b.get("avg_nll")
            if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
                failures.append({"audio": index, "candidate": candidate, "reason": "nonfinite NLL"})
                valid = False
                continue
            max_error = max(max_error, abs(x - y))
            if not math.isclose(x, y, abs_tol=atol, rel_tol=rtol):
                failures.append({"audio": index, "candidate": candidate, "reference": x, "actual": y})
        if valid:
            old_best = min(range(len(old)), key=lambda i: old[i]["avg_nll"])
            new_best = min(range(len(new)), key=lambda i: new[i]["avg_nll"])
            if old_best != new_best:
                ranking_changes.append({"audio": index, "reference_best": old_best, "actual_best": new_best})
    return {"passed": not failures and not ranking_changes, "audios": len(reference),
            "candidates": sum(map(len, reference)), "max_avg_nll_absolute_error": max_error,
            "ranking_changes": ranking_changes, "failures": failures}


async def score_prepared(engine, prepared, settings):
    with ThreadPoolExecutor(1) as executor:
        scheduler = PPLBatchScheduler(engine=engine, executor=executor, settings=settings)
        await scheduler.start()
        futures = []
        try:
            for item in prepared:
                reservation = scheduler.reserve()
                try:
                    futures.append(scheduler.submit(reservation, item))
                except BaseException:
                    scheduler.release(reservation)
                    raise
            return await asyncio.gather(*futures), scheduler.snapshot()
        finally:
            for future in futures:
                if not future.done():
                    future.cancel()
            await scheduler.stop()
            await asyncio.gather(*futures, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-scorer", type=Path, required=True,
                        help="path to the original firered_batch_ppl.py (offline validation only)")
    parser.add_argument("--no-use-gpu", action="store_true")
    parser.add_argument("--encoder-precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--decoder-precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--include-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--softmax-smoothing", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.limit <= 128 or min(args.atol, args.rtol) < 0:
        parser.error("limit must be 1..128; tolerances must be nonnegative")
    rows = []
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        wav = Path(item["wav_path"])
        waveform, sample_rate = sf.read(wav if wav.is_absolute() else args.manifest.parent / wav, dtype="int16")
        if waveform.ndim != 1 or sample_rate != 16000:
            raise ValueError("verification sources must be mono 16 kHz")
        item["pcm"] = waveform.astype("<i2", copy=False)
        item["sample_rate"] = sample_rate
        rows.append(item)
        if len(rows) == args.limit:
            break
    if not rows:
        parser.error("manifest is empty")

    spec = importlib.util.spec_from_file_location("ppl_legacy_reference", args.reference_scorer)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    reference_scorer = module.FireRedBatchPPLScorer(
        args.model_dir, "aed", firered_pkg_dir=str(REPO_ROOT), use_gpu=not args.no_use_gpu,
        use_half=False, softmax_smoothing=args.softmax_smoothing, include_eos=args.include_eos,
    )
    reference = []
    for index, row in enumerate(rows):
        scores = reference_scorer.score(
            [{"uid": str(index), "wav_path": (row["sample_rate"], row["pcm"].astype("float32")),
              "texts": row["texts"]}],
            include_details=True,
        )
        reference.append([score_fields(result) for result in scores])
    # Compare sequentially so two full checkpoints never occupy the GPU together.
    del reference_scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    settings = Settings.model_validate({
        "model": {"model_dir": args.model_dir, "include_eos": args.include_eos,
                  "softmax_smoothing": args.softmax_smoothing},
        "runtime": {"use_gpu": not args.no_use_gpu, "encoder_precision": args.encoder_precision,
                    "decoder_precision": args.decoder_precision},
    })
    engine = PPLScorer(settings)
    prepared = [prepare_pcm(engine, settings, str(i), row["pcm"].tobytes(), row["sample_rate"], row["texts"])
                for i, row in enumerate(rows)]
    actual, scheduler_stats = asyncio.run(score_prepared(engine, prepared, settings))
    report = compare(reference, actual, atol=args.atol, rtol=args.rtol)
    report.update({"settings": settings.model_dump(), "scheduler": scheduler_stats,
                   "reference": reference, "actual": actual, "scorer": engine.stats})
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"reference", "actual"}},
                     ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
