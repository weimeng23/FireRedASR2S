# FireRedLID inference acceleration handoff

Last updated: 2026-07-15

## Goal

Accelerate the original FireRedLID inference path without retraining or changing
its prediction semantics.

The intended data flow is:

```text
audio path or waveform
  -> official audio loading
  -> truncate to at most 60 seconds
  -> official FBank and CMVN
  -> batch planner (hard split, optional bucketing, padding)
  -> replaceable Encoder backend
  -> official PyTorch Transformer Decoder and beam search
  -> official tokenizer and result formatting
```

The model architecture and decoding contract must remain unchanged:

- Original FireRedLID checkpoint.
- Original Conformer Encoder weights.
- Original Transformer Decoder weights.
- Original `beam_size=3` and `decode_max_len=2` beam search.
- Original language labels and confidence calculation.
- Original Encoder return contract:
  `encoder_outputs`, `encoder_lengths`, `encoder_mask`.

The first delivery target is an embeddable Python API. A thin Triton wrapper is
deferred until the Python/TensorRT runtime is stable. The intended production
GPUs are RTX PRO 5000 and L20. Both online latency and offline throughput modes
must be benchmarked. Engines must be built and measured separately for each GPU
class.

## Repository state

Current branch and implementation base after the ONNX-layout follow-up:

```text
branch: lid-infer-accel
HEAD:   aaf9d86
remote: origin/lid-infer-accel (synchronized)
```

Committed feature stack, newest first:

```text
aaf9d86 chore: stop tracking FireRedLID superpowers docs
c41fd86 fix(fireredlid): export ONNX weights directly under data
055567b feat(fireredlid): disable dynamo in ONNX encoder export
b382a1c chore: update dependencies in pyproject.toml and remove unused fireredlid runtime configuration
40bdb41 test(fireredlid): add inference verification and benchmarks
c0838c8 feat(fireredlid): add TensorRT engine contract
0b42914 feat(fireredlid): add dynamic ONNX encoder export
35b437c feat(fireredlid): add configurable inference backends
8ebe7aa feat(fireredlid): add pluggable encoder backends
fa33c86 refactor(fireredlid): separate feature extraction from padding
a2ef7df feat(fireredlid): add physical batch planner
7b4d82f perf(fireredlid): vectorize encoder padding mask
4fa9922 docs(fireredlid): add macOS implementation plan
cdcdb35 docs(fireredlid): clarify frontend and encoder contracts
```

The Phase A snapshot above remains the starting point for the active Linux
NVIDIA plan. Mac-developable Phase B Tasks 1-5 are committed through
`9fa4b85` (`test(fireredlid): cover TensorRT tolerance propagation`). Task 6
adds the deterministic benchmark-matrix runner and this handoff update. GPU
build, correctness, and performance evidence remain a separate user-run Linux
acceptance task.

Important details:

- `FireRedLID/` contains the real checkpoint, dictionary, CMVN and config. Do
  not add it to Git.
- `runtime/fireredlid/artifacts/` is ignored and currently uses about 2.7 GiB.
- The root model directory is ignored with the anchored `/FireRedLID/` rule;
  leaving the rule unanchored also ignores lowercase `fireredlid` source
  directories on the default macOS filesystem.
- `uv.lock` exists but the user's `.gitignore` currently ignores it.
- `fireredasr2s.egg-info/` is generated packaging metadata.

## Current Progress

### Completed implementation

1. The Encoder padding mask was rewritten as vectorized tensor operations. The
   output remains equivalent to the official implementation and is exportable.
2. Feature extraction was separated from batch padding so the frontend and
   Encoder can be timed and optimized independently. FBank and CMVN remain the
   official algorithms.
3. Audio is truncated before FBank according to `max_audio_seconds`, defaulting
   to 60 seconds.
4. A physical batch planner was added:
   - arbitrary logical request batch sizes are accepted;
   - physical batches are hard-split at the configured/backend maximum;
   - `batch_strategy=none` preserves order without length reordering;
   - `batch_strategy=bucket` groups by processed duration;
   - `batch_strategy=auto` buckets only when estimated padding savings reach
     20%;
   - results are restored to original request order.
5. The Encoder backend abstraction and these implementations were added:
   - eager PyTorch;
   - `torch.compile` PyTorch;
   - TensorRT runtime contract with lazy TensorRT imports and eager fallback
     only when explicitly configured.
6. The public `FireRedLid` Python API supports backend, profile, maximum audio
   duration, batch strategy, engine directory and fallback selection.
7. Dynamic FP32 ONNX Encoder export and ONNX Runtime verification tooling were
   added.
8. TensorRT engine builder, manifest, profile loading, checkpoint/ONNX hashing,
   shape validation and serialized `.plan` runtime loading were implemented
   statically.
9. Benchmark tooling reports stage timing, P50/P95/P99, utterances/s,
   audio-seconds/s, RTF, padding ratio, physical shapes and peak CUDA memory.
10. Latency and throughput example configurations and the Linux handoff are
    documented in `runtime/fireredlid/README.md`.
11. The 99-test FireRedLID unit/contract suite covers the mask, feature
    truncation, planner, backend adapter, configuration, result ordering, ONNX
    export, verification, benchmark summary, TensorRT artifact validation, and
    benchmark orchestration.
12. Mac-developable Phase B work adds artifact preflight, strict TensorRT
    manifest/hash validation, fake-context execution coverage, sequential
    final-label parity reporting, and backend-specific verification tolerance
    reporting. These are contract/code-path results, not Linux GPU evidence.
13. `run_benchmark_matrix.py` deterministically expands each selected profile
    into eager, compile, and TensorRT commands for Encoder, model, and
    end-to-end scopes. It defaults to a non-executing dry-run, validates all
    required paths before `--execute`, fails fast, and writes a matrix index.

### Main changed files

Modified official files:

- `fireredasr2s/fireredlid/data/feat.py`: separates extraction from padding,
  truncates before FBank and retains CMVN behavior.
- `fireredasr2s/fireredlid/lid.py`: extended configuration, logical/physical
  batch planning, backend selection, fallback, stage timing and original-order
  restoration.
- `fireredasr2s/fireredlid/models/fireredlid_aed.py`: routes the Encoder through
  the selected backend while preserving all three official Encoder outputs and
  the original Decoder/beam search.
- `fireredasr2s/fireredlid/models/module/conformer_encoder.py`: vectorized mask
  generation and removal of unused intermediate-output collection.
- `.gitignore`, `pyproject.toml`, `requirements.txt`: artifact/dependency setup.

New Python runtime modules:

- `fireredasr2s/fireredlid/runtime/batch_planner.py`
- `fireredasr2s/fireredlid/runtime/encoder_backend.py`
- `fireredasr2s/fireredlid/runtime/pytorch_backend.py`
- `fireredasr2s/fireredlid/runtime/tensorrt_backend.py`

New operational tools:

- `runtime/fireredlid/export_encoder_onnx.py`
- `runtime/fireredlid/verify.py`
- `runtime/fireredlid/build_engine.py`
- `runtime/fireredlid/benchmark.py`
- `runtime/fireredlid/run_benchmark_matrix.py`
- `runtime/fireredlid/verify_labels.py`
- `runtime/fireredlid/profiles.yaml`
- `runtime/fireredlid/example_manifest.jsonl`
- `runtime/fireredlid/README.md`

Design and implementation plan:

- `docs/superpowers/specs/2026-07-14-fireredlid-inference-acceleration-design.md`
- `docs/superpowers/plans/2026-07-14-fireredlid-inference-acceleration-macos.md`
- `docs/superpowers/plans/2026-07-15-fireredlid-inference-acceleration-linux-nvidia.md`

The macOS plan is complete. The Linux NVIDIA plan is the active plan. These
`docs/superpowers` files are intentionally retained only in the local checkout
and ignored by Git.

### Validation completed before the UV update

The original Mac validation environment used Python 3.11.8, PyTorch 2.6.0,
ONNX 1.19.0 and ONNX Runtime 1.21.0.

- 28/28 FireRedLID tests passed.
- Real eager smoke labels passed:
  - `assets/hello_zh.wav` -> `zh mandarin`
  - `assets/hello_en.wav` -> `en`
- The FP32 dynamic ONNX Encoder passed 10/10 combinations:
  - durations: 1, 5, 15, 30 and 60 seconds;
  - batch sizes: 1 and 2;
  - `encoder_lengths` and `encoder_mask` exactly matched eager;
  - `encoder_outputs` passed the configured numerical tolerance.
- The old full report is
  `runtime/fireredlid/artifacts/verify.post-commit.fp32.json`.
- CPU eager latency and throughput smoke reports were generated. These used
  only two short sample files and one measured iteration, so they are smoke
  evidence rather than capacity results.

The exported FP32 ONNX bundle is about 2.7 GiB. The main ONNX file uses 548
external weight files. Copy the whole artifact directory, not only
`encoder.fp32.onnx`.

### Current UV environment and new validation

The user updated the root UV environment to:

```text
Python:            3.11.8
PyTorch:           2.10.0
NumPy:             2.4.2
kaldiio:           2.18.1
kaldi-native-fbank: 1.22.3
ONNX:              1.22.0
ONNX Runtime:      1.27.0
platform:          macOS 26.4.1 arm64
```

Validation completed in this new environment:

- `compileall` passed for `fireredasr2s/fireredlid` and
  `runtime/fireredlid`.
- Real eager smoke labels still pass after the FBank upgrade:
  - `hello_zh` -> `zh mandarin`, confidence 0.996;
  - `hello_en` -> `en`, confidence 0.996.
- PyTorch 2.10 ONNX export was repaired by explicitly selecting the legacy
  exporter with `dynamo=False`. The fresh full Encoder artifact is
  `runtime/fireredlid/artifacts/torch2.10-opset17-fp32/encoder.fp32.onnx`.
- The fresh artifact is opset 17 with exactly two inputs and three outputs,
  contains 548 external FP32 initializers, and passed all 10/10 dynamic
  duration/batch cases. Its largest observed absolute Encoder-output error was
  `2.288818359375e-05`. All external files are stored under `data/`, and every
  ONNX external-data location is relative to that directory. The exporter
  writes weights directly into `data/`; it only rewrites and promotes the small
  main ONNX graph file, so exporting does not move the 548 weight files.
- Fresh PyTorch 2.10 report:
  `runtime/fireredlid/artifacts/torch2.10-opset17-fp32/verify.fp32.json`.
- All 29/29 FireRedLID tests pass under the declared UV environment.
- New PyTorch 2.10 CPU smoke benchmark reports:
  - `runtime/fireredlid/artifacts/benchmark.uv-20260714.eager.latency.end-to-end.json`
  - `runtime/fireredlid/artifacts/benchmark.uv-20260714.eager.throughput.end-to-end.json`

The new CPU benchmark used one stable iteration and two short files. Do not
infer production performance or a PyTorch-version regression from it. Final
performance conclusions require Linux GPU runs with warm-up and representative
manifests.

### Phase B Mac code completion

Mac-developable Phase B Tasks 1-6 are implemented. The benchmark-matrix runner
can be dry-run on macOS without importing TensorRT or launching a child
benchmark. A latency dry-run expands to nine commands with batch 1, no
bucketing, five warm-ups, and 50 measured iterations; throughput expands to the
same nine backend/scope combinations with logical batch 100, automatic
batching, three warm-ups, and 20 measured iterations.

This completion is limited to orchestration and Mac-testable contracts. No
TensorRT engine, CUDA compile result, label-parity result, RTX PRO 5000/L20
timing, or matrix index from a real Linux GPU run has been produced yet.
Fresh Mac verification after Task 6 passed all 99 FireRedLID tests with the
four pre-existing legacy ONNX exporter warnings. The exact latency dry-run
printed nine commands and created no output directory.

### Not yet completed

1. A TensorRT `encoder.plan` has not been built or executed.
2. TensorRT output parity and final-label parity have not been validated.
3. CUDA `torch.compile` has not been executed or benchmarked.
4. RTX PRO 5000 and L20 latency/throughput benchmarks have not been run.
5. The initial engine profile still has a hard maximum physical batch of 4 and
   6000 FBank frames. The optimal maximum batch for each target GPU has not been
   measured.
6. Representative production LID accuracy/regression data has not been run;
   only the two repository smoke audio files have final-label checks.
7. FBank 1.15 versus 1.22.3 feature tensors have not been compared numerically.
   Only the two final smoke labels were checked after the upgrade.
8. A Triton model repository and thin wrapper have not been implemented.

## What Worked

### Stable Encoder boundary

Replacing only the Conformer Encoder keeps the high-risk parts unchanged:
official FBank/CMVN, Decoder, beam search, tokenizer and result formatting.
The backend adapter preserves all three official Encoder outputs, so the
Decoder does not need special cases for eager, compile or TensorRT.

### Vectorized mask

The tensor-only mask implementation is exact, removes a batch-level Python
loop and is exportable. It is covered by unit and `torch.export` tests.

### Optional rather than mandatory bucketing

Bucketing is not forced. Online latency defaults can preserve request order and
avoid planner overhead. Offline `auto` mode only buckets when estimated padding
savings reach 20%. Any logical batch is still hard-split when it exceeds the
backend/engine maximum physical batch.

### PyTorch 2.6 FP32 ONNX baseline

The PyTorch 2.6 legacy ONNX export produced a dynamic opset-17 Encoder with
external data. It passed all tested durations and batch sizes, and remains
compatible with ONNX Runtime 1.27.

### Mac-compatible FBank package

`kaldi-native-fbank` 1.22.3 has a macOS arm64 wheel and works for the current
two-file real inference smoke test.

## What Didn't Work

### `kaldi-native-fbank==1.15` from PyPI on Apple Silicon

PyPI version 1.15 has no macOS arm64 wheel or source distribution. `uv sync`
therefore failed on the user's Mac. The environment was changed to
`kaldi-native-fbank>=1.22.3`, which resolves successfully on arm64.

Do not restore the PyPI 1.15 requirement on Apple Silicon. If exact upstream
1.15 behavior is required, use the previously pinned Git revision or build from
source, then compare features against 1.22.3.

### Tests were absent from the default UV environment

Before dependency consolidation, `pytest` was not declared and running:

```bash
uv run python -m pytest tests/fireredlid -v
```

failed immediately with `No module named pytest`. This is now resolved:
`pytest` is in the root `dev` dependency group, `PyYAML` is a root runtime
dependency, and the duplicate `runtime/fireredlid/pyproject.toml` was removed.

### PyTorch 2.10 ONNX exporter compatibility

With only temporary `pytest`, 26 tests passed and two ONNX export tests failed
because PyTorch 2.10 imports `onnxscript`, which is not declared.

Temporarily adding `onnxscript` did not fully solve the problem. PyTorch 2.10
uses its Dynamo ONNX exporter by default, while
`runtime/fireredlid/export_encoder_onnx.py` supplies legacy `dynamic_axes`.
The resulting tiny test model unexpectedly requires a third input named
`encoder_lengths_orig`, although the runtime contract supplies only `features`
and `feature_lengths`. ONNX Runtime then fails with:

```text
Required inputs (['encoder_lengths_orig']) are missing from input feed
```

The exporter also warns that opset 17 is below its native implementation level,
so it exports as opset 18 and attempts a conversion back to 17.

Dependency-only changes did not fix this. The export API itself needed to be
made explicit. Two paths were evaluated:

1. The selected compatibility path explicitly sets `dynamo=False` and
   preserves the known-good legacy dynamic-axes/opset-17 contract.
2. Modern path: keep Dynamo export, replace `dynamic_axes` with
   `dynamic_shapes`, select an explicit supported opset (likely 18), and verify
   TensorRT 10/11 parser compatibility.

The selected path now retains exactly two inputs and three outputs, supports
dynamic batch/time, and passes the full real Encoder verification. The legacy
exporter emits a deprecation warning, so a future move to the modern path still
requires separate ONNX and TensorRT compatibility work.

### TensorRT cannot be validated on Mac

The Mac phase can validate Python contracts, ONNX and lazy failure behavior.
It cannot deserialize or execute TensorRT, run CUDA compile, or produce valid
RTX PRO 5000/L20 performance numbers.

## Next Steps

### 1. Transfer the completed Linux handoff code

Mac-testable Phase B code is complete through benchmark orchestration. Transfer
the checkout and complete artifact bundle to each Linux NVIDIA host. Keep the
GPU-only checklist open until returned engine manifests, verification reports,
label-parity reports, benchmark indexes, and individual benchmark reports have
been reviewed.

### 2. Build and verify TensorRT on Linux NVIDIA

1. Use the same software stack intended for deployment.
2. Build an engine separately on RTX PRO 5000 and L20; do not assume one plan
   is portable or optimal across GPU classes.
3. Start with the checked-in default profile:

   ```text
   features min: [1, 1, 80]
   features opt: [1, 1000, 80]
   features max: [4, 6000, 80]
   ```

4. Verify engine hashes, input/output names, shape ranges and FP16 dtype.
5. Compare TensorRT Encoder outputs to eager FP16. The current TensorRT verifier
   uses `rtol=1e-2`, `atol=1e-2`; lengths and masks must remain exact.
6. Compare final labels and confidence against eager FP16 on a representative
   dataset, not only two sample files.
7. Confirm logical batches larger than engine max batch are split and restored
   to original order.
8. Increase max batch independently on each GPU until memory, latency and
   throughput measurements identify the useful limit.

### 3. Benchmark all backends

On each target GPU compare eager FP16, compile FP16 and TensorRT FP16 using the
same audio manifest and parameters.

Run both:

- online latency: batch 1, no bucketing, warm-up followed by at least 50 stable
  iterations;
- offline throughput: heterogeneous representative audio, logical batch around
  100, `batch_strategy=auto`, at least 20 stable iterations.

Collect `encoder`, `model` and `end-to-end` scopes. Report P50/P95/P99,
utterances/s, audio-seconds/s, RTF, peak memory, actual physical shapes,
padding ratio and stage totals. Use the results to select profiles and maximum
physical batch separately for RTX PRO 5000 and L20.

### 4. Add Triton only after Python runtime acceptance

The later Triton layer should remain thin:

- Triton dynamically batches requests.
- The wrapper converts Triton tensors/metadata into this Python runtime's
  logical batch.
- The existing planner enforces engine maximum shapes and restores order.
- Use multiple model instances/execution contexts for concurrency; do not call
  one `FireRedLid` instance concurrently.

## Reproduction commands

All commands assume the repository root:

```bash
cd /Users/weimeng/workspace/FireRedASR2S
```

### Inspect the exact checkout

```bash
git branch --show-current
git rev-parse --short HEAD
git status --short
```

Expected committed snapshot before the local Phase B work is branch
`lid-infer-accel`, HEAD `aaf9d86`.

### Install and inspect the current UV environment

```bash
uv sync

uv run python -c 'import platform, torch, numpy, kaldiio, kaldi_native_fbank, onnx, onnxruntime; print(platform.platform()); print("torch", torch.__version__); print("numpy", numpy.__version__); print("kaldiio", kaldiio.__version__); print("kaldi_native_fbank", kaldi_native_fbank.__version__); print("onnx", onnx.__version__); print("onnxruntime", onnxruntime.__version__)'
```

### Reproduce the current unit-test state

Runtime/contract tests:

```bash
uv run python -m pytest tests/fireredlid -v
```

Expected Phase B result: 99 passed with four legacy ONNX exporter warnings; no
`onnxscript` dependency is required. Re-run the complete suite after copying
the Phase B commits rather than relying on the earlier 29-test Phase A
snapshot.

Compile check:

```bash
uv run python -m compileall -q \
  fireredasr2s/fireredlid runtime/fireredlid
```

### Real eager smoke test

```bash
uv run python -c 'from fireredasr2s.fireredlid.lid import FireRedLid, FireRedLidConfig; lid=FireRedLid.from_pretrained("FireRedLID", FireRedLidConfig(use_gpu=False, backend="eager", max_audio_seconds=60.0)); print(lid.process(["hello_zh","hello_en"], ["assets/hello_zh.wav","assets/hello_en.wav"]))'
```

Expected labels:

```text
hello_zh -> zh mandarin
hello_en -> en
```

### Export and verify the PyTorch 2.10 opset-17 FP32 ONNX

Use a new output directory rather than overwriting an existing artifact:

```bash
uv run python runtime/fireredlid/export_encoder_onnx.py \
  --model-dir FireRedLID \
  --output-dir runtime/fireredlid/artifacts/torch2.10-opset17-fp32

uv run python runtime/fireredlid/verify.py \
  --model-dir FireRedLID \
  --onnx runtime/fireredlid/artifacts/torch2.10-opset17-fp32/encoder.fp32.onnx \
  --report runtime/fireredlid/artifacts/torch2.10-opset17-fp32/verify.fp32.json
```

Inspect the ONNX contract without loading all external tensors:

```bash
uv run python -c 'import onnx; p="runtime/fireredlid/artifacts/torch2.10-opset17-fp32/encoder.fp32.onnx"; m=onnx.load(p, load_external_data=False); print("opsets", [(x.domain, x.version) for x in m.opset_import]); print("inputs", [(x.name, x.type.tensor_type.elem_type) for x in m.graph.input]); print("outputs", [(x.name, x.type.tensor_type.elem_type) for x in m.graph.output])'
```

### Mac CPU benchmark smoke

Latency mode processes the two examples as separate batch-1 requests:

```bash
uv run python runtime/fireredlid/benchmark.py \
  --model-dir FireRedLID \
  --manifest runtime/fireredlid/example_manifest.jsonl \
  --backend eager \
  --device cpu \
  --precision fp32 \
  --profile latency \
  --scope end-to-end \
  --warmup 0 \
  --iterations 1 \
  --output runtime/fireredlid/artifacts/benchmark.local.eager.latency.json
```

Throughput mode batches both examples:

```bash
uv run python runtime/fireredlid/benchmark.py \
  --model-dir FireRedLID \
  --manifest runtime/fireredlid/example_manifest.jsonl \
  --backend eager \
  --device cpu \
  --precision fp32 \
  --profile throughput \
  --scope end-to-end \
  --warmup 0 \
  --iterations 1 \
  --output runtime/fireredlid/artifacts/benchmark.local.eager.throughput.json
```

These commands validate the benchmark path only; they are not production
capacity tests.

### Linux TensorRT engine build and correctness verification

After installing a matching CUDA/TensorRT/PyTorch environment and copying the
complete ONNX external-data bundle:

```bash
uv run python runtime/fireredlid/build_engine.py \
  --onnx runtime/fireredlid/artifacts/torch2.10-opset17-fp32/encoder.fp32.onnx \
  --checkpoint FireRedLID/model.pth.tar \
  --profiles runtime/fireredlid/profiles.yaml \
  --output-dir runtime/fireredlid/artifacts/engine

uv run python runtime/fireredlid/verify.py \
  --model-dir FireRedLID \
  --backend tensorrt \
  --engine-dir runtime/fireredlid/artifacts/engine
```

The engine directory must contain:

```text
encoder.plan
manifest.json
profiles.yaml
```

### Linux online-latency benchmark matrix

Use the same manifest for every backend. The orchestrator generates all nine
backend/scope commands and executes them sequentially only with `--execute`:

```bash
uv run python runtime/fireredlid/run_benchmark_matrix.py \
  --model-dir FireRedLID \
  --manifest /path/to/representative-latency.jsonl \
  --engine-dir runtime/fireredlid/artifacts/engine \
  --profile latency \
  --output-dir runtime/fireredlid/artifacts/rtx-pro-5000/latency \
  --execute
```

The following loop shows the equivalent eager command expansion for manual
troubleshooting:

```bash
for scope in encoder model end-to-end; do
  uv run python runtime/fireredlid/benchmark.py \
    --model-dir FireRedLID \
    --manifest /path/to/representative-latency.jsonl \
    --backend eager \
    --device cuda \
    --precision fp16 \
    --profile latency \
    --batch-strategy none \
    --logical-batch-size 1 \
    --scope "$scope" \
    --warmup 5 \
    --iterations 50 \
    --output "runtime/fireredlid/artifacts/benchmark.eager.latency.$scope.json"
done
```

Compile substitution:

```text
--backend compile
```

TensorRT substitution:

```text
--backend tensorrt --engine-dir runtime/fireredlid/artifacts/engine
```

### Linux offline-throughput benchmark matrix

The manifest should contain a representative distribution of short and long
audio, not only two samples. Use a different output directory from latency and
from every other GPU:

```bash
uv run python runtime/fireredlid/run_benchmark_matrix.py \
  --model-dir FireRedLID \
  --manifest /path/to/representative-throughput.jsonl \
  --engine-dir runtime/fireredlid/artifacts/engine \
  --profile throughput \
  --output-dir runtime/fireredlid/artifacts/rtx-pro-5000/throughput \
  --execute
```

The equivalent TensorRT-only expansion is:

```bash
for scope in encoder model end-to-end; do
  uv run python runtime/fireredlid/benchmark.py \
    --model-dir FireRedLID \
    --manifest /path/to/representative-throughput.jsonl \
    --backend tensorrt \
    --engine-dir runtime/fireredlid/artifacts/engine \
    --device cuda \
    --precision fp16 \
    --profile throughput \
    --batch-strategy auto \
    --logical-batch-size 100 \
    --scope "$scope" \
    --warmup 3 \
    --iterations 20 \
    --output "runtime/fireredlid/artifacts/benchmark.tensorrt.throughput.$scope.json"
done
```

Repeat with eager and compile using identical inputs, scope, warm-up and
iteration settings. The logical batch may be 100 even when the engine physical
maximum is 4; the planner splits it into physical sub-batches. Tune the engine
maximum only after recording memory and latency/throughput on each target GPU.

Repeat both matrix invocations on L20 using its own engine and output
directories under `runtime/fireredlid/artifacts/l20/`. Each successful matrix
writes nine benchmark JSON files and `matrix.index.json`. Return the index and
reports for review; a Mac dry-run is command-generation evidence only.

## Key cautions for the next agent

- Start by reading this file and checking the live diff; do not assume the
  working tree is clean.
- Preserve the user's root dependency and `.gitignore` edits unless explicitly
  asked to change them.
- Do not commit the checkpoint, ONNX external data, reports or TensorRT plan.
- Do not claim TensorRT support is complete based only on builder code and Mac
  contract tests.
- Do not compare Mac CPU smoke timing with Linux GPU latency.
- Do not compare latency batch 1 against throughput batch 100 as if they were
  the same workload.
- Do not change Decoder, beam size, tokenizer, FBank or CMVN to obtain speedups
  without explicit approval.
- Build a different TensorRT engine for each target GPU/software stack.
