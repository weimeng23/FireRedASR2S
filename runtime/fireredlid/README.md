# FireRedLID inference runtime

This directory contains the export, verification, TensorRT build, and benchmark
tools for the original FireRedLID model. The Conformer Encoder backend is
replaceable; FBank, CMVN, the Transformer Decoder, `beam_size=3`, and the
two-step beam search remain unchanged.

The public Python entry point remains:

```python
from fireredasr2s.fireredlid.lid import FireRedLid, FireRedLidConfig

config = FireRedLidConfig(
    use_gpu=False,
    backend="eager",
    profile="latency",
    max_audio_seconds=60.0,
)
lid = FireRedLid.from_pretrained("FireRedLID", config)
results = lid.process(["hello_zh"], ["assets/hello_zh.wav"])
```

Every Encoder backend preserves the official three-value contract:
`encoder_outputs`, `encoder_lengths`, and `encoder_mask`.

## macOS validation

macOS can validate CPU eager inference, batching behavior, dynamic FP32 ONNX
export, ONNX Runtime numerical parity, and the TensorRT artifact contract. It
cannot build or run a TensorRT engine because TensorRT requires Linux and an
NVIDIA GPU.

Install the exact official `kaldi-native-fbank` v1.15 source revision declared
by the repository, then run:

```bash
python3 -m pip install "git+https://github.com/csukuangfj/kaldi-native-fbank.git@f68c6b43f739697d7ab02ff6debacee130e1d541"
python3 -m pytest tests/fireredlid -v
python3 -m compileall -q fireredasr2s/fireredlid runtime/fireredlid
```

Export and verify the dynamic ONNX Encoder:

```bash
python3 runtime/fireredlid/export_encoder_onnx.py \
  --model-dir FireRedLID \
  --output-dir runtime/fireredlid/artifacts/torch2.10-opset17-fp32

python3 runtime/fireredlid/verify.py \
  --model-dir FireRedLID \
  --onnx runtime/fireredlid/artifacts/torch2.10-opset17-fp32/encoder.fp32.onnx \
  --report runtime/fireredlid/artifacts/torch2.10-opset17-fp32/verify.fp32.json
```

Encoder verification uses backend-specific numerical tolerances: ONNX defaults
to `--rtol 0.001 --atol 0.0001`, while TensorRT defaults to
`--rtol 0.02 --atol 0.02`. Pass either option explicitly to override its
backend default. The JSON report records both the CLI values and the resolved
tolerances under `arguments`, along with top-level `environment`, `cases`, and
`passed` fields. `passed` is false when any case fails; existing case-level
timing, shape, status, and `max_abs_error` fields remain available.

Generated artifacts are ignored by Git. A large exported model may use ONNX
external-data files, so copy the complete artifact directory rather than only
`encoder.fp32.onnx`. External weight files are stored under the artifact's
`data/` subdirectory. The exporter writes the weights directly into `data/`;
it only promotes the small main ONNX graph file to the artifact root after
updating its relative external-data paths.

Run a CPU end-to-end smoke benchmark:

```bash
python3 runtime/fireredlid/benchmark.py \
  --model-dir FireRedLID \
  --manifest runtime/fireredlid/example_manifest.jsonl \
  --backend eager \
  --device cpu \
  --precision fp32 \
  --profile latency \
  --scope end-to-end \
  --warmup 0 \
  --iterations 1
```

## Online latency configuration

Use one logical request at a time, avoid internal reordering, and prefer the
compile latency mode or the FP16 TensorRT engine on Linux NVIDIA:

```python
config = FireRedLidConfig(
    use_gpu=True,
    use_half=True,
    backend="tensorrt",
    engine_dir="runtime/fireredlid/artifacts/rtx-pro-5000/engine",
    profile="latency",
    batch_strategy="none",
    max_audio_seconds=60.0,
)
```

Benchmark it with:

```bash
python3 runtime/fireredlid/benchmark.py \
  --model-dir FireRedLID \
  --manifest runtime/fireredlid/example_manifest.jsonl \
  --backend tensorrt \
  --engine-dir runtime/fireredlid/artifacts/rtx-pro-5000/engine \
  --device cuda \
  --precision fp16 \
  --profile latency \
  --batch-strategy none \
  --logical-batch-size 1 \
  --scope end-to-end \
  --warmup 5 \
  --iterations 50
```

The benchmark reports P50/P95/P99 latency, utterances/s, audio-seconds/s, RTF,
peak allocated GPU memory, actual physical input shapes, padding ratio, and
stage timings. CUDA synchronization is deliberately enabled around measured
stages, so use these results for attribution and compare all backends using the
same command.

## Offline throughput configuration

For a heterogeneous logical batch, `auto` planning keeps the batch intact when
padding waste is small and physically groups inputs only when the estimated
padding saving reaches 20%. TensorRT's profile still imposes a hard physical
maximum of batch 4 and 6000 FBank frames in the checked-in initial profile.

```python
config = FireRedLidConfig(
    use_gpu=True,
    use_half=True,
    backend="tensorrt",
    engine_dir="runtime/fireredlid/artifacts/rtx-pro-5000/engine",
    profile="throughput",
    batch_strategy="auto",
    max_audio_seconds=60.0,
)
```

Benchmark a larger manifest with:

```bash
python3 runtime/fireredlid/benchmark.py \
  --model-dir FireRedLID \
  --manifest /path/to/benchmark.jsonl \
  --backend tensorrt \
  --engine-dir runtime/fireredlid/artifacts/rtx-pro-5000/engine \
  --device cuda \
  --precision fp16 \
  --profile throughput \
  --batch-strategy auto \
  --logical-batch-size 100 \
  --scope end-to-end \
  --warmup 3 \
  --iterations 20
```

The Python API accepts any logical batch size. It splits physical batches at
the smaller of `max_sub_batch_size` and the engine profile's maximum batch. A
future thin Triton wrapper can let Triton form dynamic logical batches before
calling this same runtime; the TensorRT engine's physical shape limits still
apply.

## Linux NVIDIA handoff

Copy the complete ONNX artifact directory, the model checkpoint, and this
repository to the target RTX PRO 5000 or L20 host. Keep one engine directory
per GPU class and TensorRT/CUDA software stack. The commands below preflight,
build, and verify the RTX PRO 5000 engine in its dedicated directory:

```bash
python3 runtime/fireredlid/build_engine.py \
  --onnx runtime/fireredlid/artifacts/torch2.10-opset17-fp32/encoder.fp32.onnx \
  --checkpoint FireRedLID/model.pth.tar \
  --profiles runtime/fireredlid/profiles.yaml \
  --output-dir runtime/fireredlid/artifacts/rtx-pro-5000/engine \
  --preflight-only \
  --report runtime/fireredlid/artifacts/rtx-pro-5000.engine.preflight.json

python3 runtime/fireredlid/build_engine.py \
  --onnx runtime/fireredlid/artifacts/torch2.10-opset17-fp32/encoder.fp32.onnx \
  --checkpoint FireRedLID/model.pth.tar \
  --profiles runtime/fireredlid/profiles.yaml \
  --output-dir runtime/fireredlid/artifacts/rtx-pro-5000/engine

python3 runtime/fireredlid/verify.py \
  --model-dir FireRedLID \
  --backend tensorrt \
  --engine-dir runtime/fireredlid/artifacts/rtx-pro-5000/engine \
  --rtol 0.02 \
  --atol 0.02 \
  --report runtime/fireredlid/artifacts/rtx-pro-5000/engine/verify.encoder.fp16.json
```

Run final-label parity once for compile and once for TensorRT against the same
eager-FP16 baseline and representative manifest:

```bash
uv run python runtime/fireredlid/verify_labels.py \
  --model-dir FireRedLID \
  --manifest /path/to/representative-lid.jsonl \
  --candidate-backend compile \
  --confidence-atol 0.005 \
  --report runtime/fireredlid/artifacts/rtx-pro-5000/verify.labels.compile.json

uv run python runtime/fireredlid/verify_labels.py \
  --model-dir FireRedLID \
  --manifest /path/to/representative-lid.jsonl \
  --candidate-backend tensorrt \
  --engine-dir runtime/fireredlid/artifacts/rtx-pro-5000/engine \
  --confidence-atol 0.005 \
  --report runtime/fireredlid/artifacts/rtx-pro-5000/engine/verify.labels.json
```

The engine directory contains:

```text
encoder.plan
manifest.json
profiles.yaml
```

`encoder.plan` is TensorRT's serialized engine. `manifest.json` records tensor
contracts, profiles, environment versions, and checkpoint/ONNX hashes so a
runtime cannot silently use an incompatible engine.

Run the benchmark matrix separately for latency and throughput on each GPU.
Keep every GPU/profile pair in a distinct output directory so deterministic
report names are not overwritten. For example, run latency on RTX PRO 5000:

```bash
uv run python runtime/fireredlid/run_benchmark_matrix.py \
  --model-dir FireRedLID \
  --manifest /path/to/representative-latency.jsonl \
  --engine-dir runtime/fireredlid/artifacts/rtx-pro-5000/engine \
  --profile latency \
  --output-dir runtime/fireredlid/artifacts/rtx-pro-5000/latency \
  --execute
```

Then run throughput with its representative heterogeneous manifest and a
different directory:

```bash
uv run python runtime/fireredlid/run_benchmark_matrix.py \
  --model-dir FireRedLID \
  --manifest /path/to/representative-throughput.jsonl \
  --engine-dir runtime/fireredlid/artifacts/rtx-pro-5000/engine \
  --profile throughput \
  --output-dir runtime/fireredlid/artifacts/rtx-pro-5000/throughput \
  --execute
```

Repeat the entire preflight, build, Encoder verification, compile/TensorRT
label-parity, latency, and throughput block on L20. Replace every
`rtx-pro-5000` artifact or report path above with `l20`, including the preflight
report, engine directory, label reports, and both matrix output directories.
Each matrix invocation runs eager, compile, and TensorRT for the `encoder`,
`model`, and `end-to-end` scopes. Latency fixes logical batch size 1, no
bucketing, five warm-ups, and 50 measured iterations. Throughput fixes logical
batch size 100, automatic batching, three warm-ups, and 20 measured iterations.

Without `--execute`, the runner only prints nine shell-escaped commands. The
default matrix includes TensorRT, so `--engine-dir` is still required for a
dry-run, but that directory need not exist yet. Execution validates the model
directory, manifest file, and TensorRT engine directory before launching any
command. It stops on the first non-zero return code and records attempted
commands in `<output-dir>/matrix.index.json`. The index includes each command's
argument list, output path, and return code plus the environment and Git commit
hash; successful benchmark processes write one
`benchmark.<backend>.<profile>.<scope>.json` report each.

Run eager FP16, compile FP16, and TensorRT FP16 with identical benchmark inputs
and parameters. Collect separate reports for `--scope encoder`, `--scope model`,
and `--scope end-to-end`, and for both `--profile latency` and
`--profile throughput`. Do not compare a latency run at batch 1 with a
throughput run using a larger logical batch.

Before production use, validate final language labels against eager FP16 on a
representative LID dataset, in addition to checking Encoder numerical error.
TensorRT deserialization, CUDA execution, CUDA `torch.compile`, RTX PRO 5000/L20
performance, and final-label parity are intentionally not claimed by the macOS
work.

## Concurrency

A `FireRedLid` instance owns mutable feature-extractor state and, for TensorRT,
one execution context. Do not call the same instance concurrently. Use one
instance per service worker or serialize access. A Triton deployment should use
multiple model instances for concurrency rather than sharing one TensorRT
context across requests.

## FastAPI server

The FastAPI process loads one model instance and serializes calls to that
instance. It accepts one or more 16 kHz mono WAV files as base64 strings in a
single logical batch. It does not combine separate HTTP requests into a dynamic
batch. Audio longer than `--max-audio-seconds`, oversized encoded audio, and
oversized request bodies with a `Content-Length` header are rejected with HTTP
413 before model inference. Configure the same body-size limit in the reverse
proxy to cover chunked transfer encoding.

Start the eager backend on GPU:

```bash
uv run fireredlid-server \
  --model-dir FireRedLID \
  --backend eager \
  --use-gpu \
  --port 8000
```

For CPU development, replace `--use-gpu` with `--no-use-gpu`. Start compile by
using `--backend compile`. Start TensorRT with the required FP16 and engine
arguments:

```bash
uv run fireredlid-server \
  --model-dir FireRedLID \
  --backend tensorrt \
  --use-gpu \
  --use-half \
  --engine-dir runtime/fireredlid/artifacts/l20/engine
```

Check readiness:

```bash
curl http://127.0.0.1:8000/healthz
```

Send one 16 kHz mono WAV file to the server. Use `--repeat 3` to separate
first-request CUDA initialization from steady-state latency:

```bash
uv run python runtime/fireredlid/client.py /path/to/test.wav \
  --url http://127.0.0.1:8000/v1/lid \
  --uttid test \
  --repeat 3
```

Create a JSON request and run inference:

```bash
uv run python - <<'PY'
import base64
import json

with open("assets/hello_en.wav", "rb") as source:
    audio = base64.b64encode(source.read()).decode("ascii")

with open("/tmp/lid-request.json", "w") as output:
    json.dump(
        {"inputs": [{"uttid": "hello-en", "audio_base64": audio}]},
        output,
    )
PY

curl -X POST http://127.0.0.1:8000/v1/lid \
  -H 'content-type: application/json' \
  --data-binary @/tmp/lid-request.json
```

The server intentionally starts one Uvicorn worker. Starting multiple workers
would load multiple model copies and consume GPU memory independently.
Requests wait on an asynchronous single-inference gate before entering the
worker thread pool, so queued inference does not consume all worker threads or
delay `/healthz`.
