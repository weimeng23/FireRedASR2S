# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Yan Jia)

import logging
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from .data.feat import FeatExtractor
from .models.fireredlid_aed import FireRedLidAed
from .models.param import count_model_parameters
from .runtime.batch_planner import pad_features
from .runtime.encoder_backend import CompatibleEncoderAdapter
from .runtime.pytorch_backend import CompileEncoderBackend
from .tokenizer.lid_tokenizer import LidTokenizer


logger = logging.getLogger(__name__)


PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class FireRedLidConfig:
    use_gpu: bool = True
    use_half: bool | None = None
    encoder_precision: str | None = None
    decoder_precision: str | None = None
    backend: str = "eager"
    profile: str = "latency"
    max_audio_seconds: float = 60.0
    engine_dir: str | None = None
    fallback_backend: str | None = None
    return_diagnostics: bool = False
    beam_size: int = field(init=False, default=3)
    nbest: int = field(init=False, default=1)
    decode_max_len: int = field(init=False, default=2)
    softmax_smoothing: float = field(init=False, default=1.25)
    aed_length_penalty: float = field(init=False, default=0.6)
    eos_penalty: float = field(init=False, default=1.0)

    def __post_init__(self):
        if self.use_half is not None:
            if (
                self.encoder_precision is not None
                or self.decoder_precision is not None
            ):
                raise ValueError(
                    "use_half conflicts with explicit precision settings"
                )
            precision = "fp16" if self.use_half else "fp32"
            self.encoder_precision = precision
            self.decoder_precision = precision
        else:
            self.encoder_precision = self.encoder_precision or "fp32"
            self.decoder_precision = self.decoder_precision or "fp32"
        valid_precisions = {"fp32", "fp16", "bf16"}
        if self.encoder_precision not in valid_precisions:
            raise ValueError(
                f"unsupported encoder precision: {self.encoder_precision}"
            )
        if self.decoder_precision not in valid_precisions:
            raise ValueError(
                f"unsupported decoder precision: {self.decoder_precision}"
            )
        if self.backend not in {"eager", "compile", "tensorrt"}:
            raise ValueError(f"unsupported backend: {self.backend}")
        if self.profile not in {"latency", "throughput"}:
            raise ValueError(f"unsupported profile: {self.profile}")
        if self.max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be positive")
        if self.fallback_backend not in {None, "eager"}:
            raise ValueError("fallback_backend must be None or 'eager'")
        if self.backend == "tensorrt":
            if (
                not self.use_gpu
                or self.encoder_precision != "fp16"
                or not self.engine_dir
            ):
                raise ValueError(
                    "tensorrt requires use_gpu=True, "
                    "encoder_precision='fp16', and engine_dir"
                )


class FireRedLid:
    @classmethod
    def from_pretrained(cls, model_dir, config=FireRedLidConfig()):
        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        feat_extractor = FeatExtractor(cmvn_path)

        model_path = os.path.join(model_dir, "model.pth.tar")
        dict_path =os.path.join(model_dir, "dict.txt")
        model = load_fireredlid_model(model_path)
        tokenizer = LidTokenizer(dict_path)

        count_model_parameters(model)
        model.eval()
        return cls(
            feat_extractor,
            model,
            tokenizer,
            config,
            model_path=model_path,
        )

    def __init__(
        self,
        feat_extractor,
        model,
        tokenizer,
        config,
        model_path=None,
    ):
        self.feat_extractor = feat_extractor
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_path = model_path
        self.stage_recorder = None
        self.encoder_dtype = PRECISION_DTYPES[
            self.config.encoder_precision
        ]
        self.decoder_dtype = PRECISION_DTYPES[
            self.config.decoder_precision
        ]
        if self.config.use_gpu:
            if (
                torch.bfloat16
                in {self.encoder_dtype, self.decoder_dtype}
                and not torch.cuda.is_bf16_supported()
            ):
                raise RuntimeError(
                    "BF16 is not supported by the selected CUDA device"
                )
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.encoder.to(dtype=self.encoder_dtype)
        self.model.lid_decoder.to(dtype=self.decoder_dtype)
        self._configure_encoder_backend()

    def _configure_encoder_backend(self):
        self.backend_max_batch = None
        self.active_backend = "eager"
        if self.config.backend == "eager":
            return
        try:
            if self.config.backend == "compile":
                backend = CompileEncoderBackend(
                    self.model.encoder,
                    self.config.profile,
                )
            else:
                from .runtime.tensorrt_backend import TensorRTEncoderBackend

                backend = TensorRTEncoderBackend(
                    self.config.engine_dir,
                    checkpoint_path=self.model_path,
                )
                self.backend_max_batch = backend.max_batch
        except Exception as error:
            if self.config.fallback_backend != "eager":
                raise RuntimeError(
                    f"failed to initialize {self.config.backend} backend"
                ) from error
            logger.warning(
                "failed to initialize %s backend; using eager because "
                "fallback_backend='eager': %s",
                self.config.backend,
                error,
            )
            return
        self.model.encoder = CompatibleEncoderAdapter(backend)
        self.active_backend = self.config.backend

    def _measure_stage(self, name):
        recorder = getattr(self, "stage_recorder", None)
        if recorder is None:
            return nullcontext()
        return recorder.measure(name)

    def _infer_items(self, items):
        if (
            self.backend_max_batch is not None
            and len(items) > self.backend_max_batch
        ):
            raise ValueError(
                f"batch size {len(items)} exceeds backend maximum batch size "
                f"{self.backend_max_batch}"
            )
        with self._measure_stage("h2d"):
            features = pad_features([item.feature for item in items])
            lengths = torch.tensor(
                [item.feature.size(0) for item in items],
                dtype=torch.long,
            )
            if self.config.use_gpu:
                features = features.cuda()
                lengths = lengths.cuda()
            features = features.to(dtype=self.encoder_dtype)
        start_time = time.time()
        process_args = (
            features,
            lengths,
            self.config.beam_size,
            self.config.nbest,
            self.config.decode_max_len,
            self.config.softmax_smoothing,
            self.config.aed_length_penalty,
            self.config.eos_penalty,
        )
        recorder = getattr(self, "stage_recorder", None)
        if recorder is None:
            hypotheses = self.model.process(*process_args)
        else:
            hypotheses = self.model.process(
                *process_args,
                stage_recorder=recorder,
            )
        inference_elapsed = time.time() - start_time
        if len(hypotheses) != len(items):
            raise RuntimeError(
                "model result count does not match physical batch size"
            )
        raw_results = {}
        with self._measure_stage("result_formatting"):
            for item, hypotheses_for_item in zip(
                items, hypotheses, strict=True
            ):
                hypothesis = hypotheses_for_item[0]
                ids = [
                    int(token_id)
                    for token_id in hypothesis["yseq"].cpu()
                ]
                result = {
                    "uttid": item.uttid,
                    "lang": self.tokenizer.detokenize(ids),
                    "confidence": round(
                        hypothesis["confidence"].cpu().item(), 3
                    ),
                    "dur_s": round(item.duration_s, 3),
                }
                if isinstance(item.wav_input, str):
                    result["wav"] = item.wav_input
                if self.config.return_diagnostics:
                    result.update(
                        {
                            "backend": self.active_backend,
                            "truncated": item.truncated,
                            "processed_dur_s": round(
                                item.processed_duration_s, 3
                            ),
                        }
                    )
                raw_results[item.index] = result
        total_duration = sum(item.duration_s for item in items)
        rtf = (
            inference_elapsed / total_duration
            if total_duration > 0
            else 0.0
        )
        ordered = []
        for item in sorted(items, key=lambda value: value.index):
            result = raw_results[item.index]
            result["rtf"] = f"{rtf:.4f}"
            ordered.append(result)
        return ordered

    @torch.no_grad()
    def process_features(self, items):
        if not items:
            return []
        return self._infer_items(items)

    @torch.no_grad()
    def process(self, batch_uttid, batch_wav_path):
        with self._measure_stage("fbank"):
            items = self.feat_extractor.extract_many(
                batch_wav_path,
                batch_uttid,
                max_audio_seconds=self.config.max_audio_seconds,
            )
        return self.process_features(items)


def load_fireredlid_model(model_path):
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    #print(package["args"])
    model = FireRedLidAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    return model
