# FireRedASR2-AED PPL service

This service scores audio-conditioned candidate text using teacher forcing. It
accepts concurrent requests, batches audio across requests, and separately batches
candidates by token length under padded-work budgets. Each audio is encoded once
in the normal path. CPU preprocessing uses a separate worker pool; one GPU worker
owns one encoder/decoder pair. It does not depend on speechflow or Ray.

The first implementation supports **AED only**, using eager PyTorch. LLM, TensorRT,
shared cross-attention K/V projections and cross-request audio caches are not
implemented. GPU throughput must be measured with the actual checkpoint and GPU.

## Start

From the FireRedASR2S repository root:

```bash
bash runtime/fireredppl/start_server.sh --model-dir /models/FireRedASR2-AED
```

Or, after `uv sync`:

```bash
uv run fireredppl-server --config configs/fireredppl_server.yaml \
  --model-dir /models/FireRedASR2-AED
```

The model directory must contain `model.pth.tar`, `cmvn.ark`, `dict.txt`, and
`train_bpe1000.model`. A CPU check can use `--no-use-gpu` with both precisions at
`fp32`. Both encoder and decoder default to FP16. Encoder and decoder
precision can be selected independently (`fp32`, `fp16`, `bf16`); compare NLL,
finite-score rate and candidate ranking before deploying lower precision.

Start one process per GPU, with a separate port and `CUDA_VISIBLE_DEVICES` value.
Route callers through your existing load balancer/service discovery. Increasing
HTTP concurrency does not create additional model instances. Do not increase
Uvicorn workers as a substitute for dynamic batching.

## API and pipeline integration

`POST /score` accepts **raw little-endian signed int16 PCM**, with no WAV header.
The complete request body is limited to 16 MiB by `server.max_request_bytes`.
It uses multipart to carry binary samples and metadata together:

| Field | Value |
| --- | --- |
| `pcm` | Binary samples, `application/octet-stream` |
| `sample_rate` | `16000` |
| `channels` | `1` |
| `sample_format` | `s16le` |
| `uid` | Audio identifier |
| `candidates` | JSON string array |

For a file-based example, read the source WAV on the client once and write PCM:

```python
import soundfile as sf

pcm, sr = sf.read("sample.wav", dtype="int16")
assert sr == 16000 and pcm.ndim == 1
pcm.astype("<i2").tofile("sample.s16le")
```

The pipeline reads int16 PCM and casts it to float32 before local inference.
The API client converts those integer-valued float32 samples losslessly back to
little-endian int16 for transmission, without adding a WAV header.

```bash
curl http://127.0.0.1:12345/score \
  -F 'pcm=@sample.s16le;type=application/octet-stream' \
  -F sample_rate=16000 -F channels=1 -F sample_format=s16le \
  -F uid=sample-1 \
  -F 'candidates=["你好","您好"]'
```

Samples use the original signed int16 scale (-32768 to 32767), **not normalized
to [-1, 1]**. The server uses `numpy.frombuffer(..., dtype="<i2").astype(np.float32)`
before FBank/CMVN, matching local Actor input. The client rejects nonfinite,
out-of-range or fractional samples rather than silently clipping/rounding them.
Empty PCM and incomplete int16 samples are rejected. The server does no WAV parsing.
The legacy `decode_workers` setting now controls PCM validation, FBank/CMVN and
tokenization workers. `python-multipart` parses only the HTTP envelope and metadata.

Overlength audio/text is rejected, never silently truncated. The response preserves
candidate order and duplicates, including when unrelated requests reuse a UID:

```json
{"uid":"sample-1","scorer_type":"aed","results":[
  {"uid":"sample-1","text":"你好","ppl":2.0,"avg_nll":0.693147,"total_nll":2.079442,"token_count":3},
  {"uid":"sample-1","text":"您好","ppl":3.0,"avg_nll":1.098612,"total_nll":3.295837,"token_count":3}
]}
```

Numbers above are illustrative. EOS inclusion, tokenizer behavior and
`softmax_smoothing` follow the original AED scorer. PPL is `exp(min(avg_nll, 80))`,
rounded to six decimals. Nonfinite scores return `ppl: null` and null NLL fields;
casting an already nonfinite tensor to FP32 cannot repair it. Empty text can be
scored as EOS when enabled; text with no scored tokens is rejected otherwise.

Pipeline configuration (no business-flow changes required):

```yaml
firered_ppl:
  backend: api
  api_service: http://ppl-host:12345
  api_audio_format: pcm_s16le
  api_timeout: 35.0
```

Set `api_audio_format: pcm_s16le` explicitly when connecting the updated pipeline
client to this server. The client's default `wav` mode is retained for existing
remote WAV services; this new server does not accept the old WAV upload protocol.
In PCM mode, pass `wav_path=(sample_rate, pcm_array)` to the client, as the pipeline
already does. Raw bytes without sample metadata and filesystem paths are not
accepted by that client mode.

`polaris://...` also remains usable through the client when the service
is registered externally. The server itself does not register with Polaris.
Model precision/EOS/smoothing are now configured **on the server**; the existing
client does not send those settings. Keep the API timeout greater than the server
deadline. Retain the existing local Actor path for baseline comparisons/rollback.

Status codes: `400` invalid audio, `413` request/audio/text/work budget exceeded,
`415` unsupported content type, `422` malformed fields, `429` admission full,
`503` unavailable worker or unrecoverable single-item OOM, `504` request deadline.
The existing client retries errors; avoid excessive retries against a saturated
service. Queued cancelled/timed-out requests are discarded. In-flight work keeps
its admission capacity until its CPU/GPU operation actually finishes.

`GET /healthz` checks process liveness. `GET /readyz` checks worker health and
returns precision metadata. `GET /stats` reports queue/workload counts, actual
encoder/decoder batch calls, merged request counts, queue wait and inference time.
An unexpected inference error makes readiness fail and requires a process restart.

## Scheduling and validation

The oldest request anchors each audio bucket. A batch runs when full, when its
work budget fills, or after `max_batch_delay_ms`; sparse traffic does not wait
indefinitely. The delay is a batching window, not a bound on GPU queueing latency.
The audio buckets allow 32/16/8/2 audios for 5/15/30/60 seconds. Encoder budgets
are 24,000 padded frames and 8,000,000 attention elements, covering these full
batches at approximately 100 feature frames/s and 4x subsampling. Decoder limits
are 32 candidates, 4,096 padded tokens and 8,000,000 attention elements: 32
candidates of up to 128 tokens fit even with 60-second audio; longer texts are
split further. Validate memory and throughput on the target GPU.

These budgets are work limits, not GPU memory guarantees:

- Encoder limits use `audio_count * max_feature_frames` and the square of the
  subsampled sequence length.
- Decoder limits use candidate count, `count * max_tokens`, and
  `count * (max_tokens² + max_tokens * max_encoder_length)`.
- Queue admission includes preprocessing, pending, and in-flight requests; candidate
  and token limits account for their actual submitted work.
- OOM splits batches recursively; a single candidate that still does not fit fails.

For target-GPU testing, prepare a JSONL manifest with actual workload distributions:

```json
{"uid":"sample-1","wav_path":"sample.wav","texts":["你好","您好"]}
```

Paths are resolved relative to the manifest. The benchmark reads mono 16 kHz
PCM16 WAV sources locally before timing, and transmits only raw int16 PCM. Run:

```bash
python runtime/fireredppl/benchmark.py --url http://127.0.0.1:12345/score \
  --manifest /data/ppl_samples.jsonl --concurrency 1 8 16 32 \
  --repeats 5 --output /tmp/ppl_benchmark.json
```

The benchmark reports successful requests/s, audio seconds/s, candidates/s,
P50/P95/P99 latency, and failures. HTTP 200 with null/invalid PPL is a failure.
Use `/stats` before/after to verify actual cross-request batching. Compare the
same data against the original local scorer, including NLL, finite-score rate and
candidate ranking. Tune pipeline PPL concurrency only after checking these results.

An offline comparison command is included. Supply the original scorer source
explicitly; this dependency is used only by the verification tool, not the service:

```bash
uv run python runtime/fireredppl/verify.py \
  --model-dir /models/FireRedASR2-AED --manifest /data/ppl_samples.jsonl \
  --reference-scorer /path/to/speechflow/stages/stage06/firered_batch_ppl.py \
  --limit 32 --output /tmp/ppl_verify.json
```

This compares the original FP32 scoring calculation against the service scheduler
and scorer, loading one checkpoint at a time. Both use this repository's model
implementation/dependencies. It checks token counts, NLL tolerances and exact
best-candidate indices; ranking changes (including close ties) are reported as
failures for inspection. Set `--encoder-precision`/`--decoder-precision` to test
lower precision after the FP32 baseline passes.

## Container

Build from the repository root (a CUDA-capable Linux host is required to run GPU inference):

```bash
docker build -f runtime/fireredppl/Dockerfile -t fireredppl .
docker run --rm --gpus '"device=0"' -p 12345:12345 \
  -v /path/to/FireRedASR2-AED:/opt/FireRedASR2-AED:ro fireredppl
```

The default `CMD` reads the model from `/opt/FireRedASR2-AED`. To use a different
mount point, override the whole `CMD`: append `--model-dir /your/path` after the
image name.

This image has its own entrypoint/configuration and does not change LID deployment.
It follows the repository's existing policy of resolving dependencies at image
build time (`uv.lock` is excluded by the root `.dockerignore`).

The PPL image uses PyTorch/torchaudio 2.10.0 cu126 for V100 (Volta) support.
During the build, it switches the PyTorch index in the copied `pyproject.toml`
to cu126 and verifies the installed CUDA wheel versions. The repository metadata
and LID image are unaffected. Use a Linux x86_64 host with an NVIDIA driver
compatible with CUDA 12.6 and NVIDIA Container Toolkit.

The base image is plain `ubuntu:24.04`: the cu126 wheels bundle the CUDA runtime
and cuDNN, and `libcuda.so` comes from the host at run time, so a CUDA base image
would only add a redundant second copy. This drops about 2.1 GB of compressed
layers compared with `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04`.
