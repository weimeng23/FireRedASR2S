# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Yan Jia)

import logging
import os
import re
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from .data.feat import FeatExtractor
from .models.fireredlid_aed import FireRedLidAed
from .models.param import count_model_parameters
from .runtime.batch_planner import BatchPlanner
from .runtime.encoder_backend import CompatibleEncoderAdapter
from .runtime.pytorch_backend import CompileEncoderBackend
from .tokenizer.lid_tokenizer import LidTokenizer


logger = logging.getLogger(__name__)


@dataclass
class FireRedLidConfig:
    use_gpu: bool = True
    use_half: bool = False
    backend: str = "eager"
    profile: str = "latency"
    max_audio_seconds: float = 60.0
    batch_strategy: str | None = None
    engine_dir: str | None = None
    max_sub_batch_size: int | None = None
    fallback_backend: str | None = None
    return_diagnostics: bool = False
    beam_size: int = field(init=False, default=3)
    nbest: int = field(init=False, default=1)
    decode_max_len: int = field(init=False, default=2)
    softmax_smoothing: float = field(init=False, default=1.25)
    aed_length_penalty: float = field(init=False, default=0.6)
    eos_penalty: float = field(init=False, default=1.0)

    def __post_init__(self):
        if self.backend not in {"eager", "compile", "tensorrt"}:
            raise ValueError(f"unsupported backend: {self.backend}")
        if self.profile not in {"latency", "throughput"}:
            raise ValueError(f"unsupported profile: {self.profile}")
        if self.max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be positive")
        if self.fallback_backend not in {None, "eager"}:
            raise ValueError("fallback_backend must be None or 'eager'")
        if self.backend == "tensorrt":
            if not self.use_gpu or not self.use_half or not self.engine_dir:
                raise ValueError(
                    "tensorrt requires use_gpu=True, use_half=True, "
                    "and engine_dir"
                )

    @property
    def resolved_batch_strategy(self):
        if self.batch_strategy is not None:
            return self.batch_strategy
        return "none" if self.profile == "latency" else "auto"


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
        self._configure_encoder_backend()
        if self.config.use_gpu:
            if self.config.use_half:
                self.model.half()
            self.model.cuda()
        else:
            self.model.cpu()

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
        limits = [
            value
            for value in (
                self.config.max_sub_batch_size,
                self.backend_max_batch,
            )
            if value is not None
        ]
        max_batch = min(limits) if limits else None
        planner = BatchPlanner(
            strategy=self.config.resolved_batch_strategy,
            max_sub_batch_size=max_batch,
        )
        raw_results = {}
        inference_elapsed = 0.0
        for planned in planner.plan(items):
            with self._measure_stage("h2d"):
                features = planned.padded_features
                lengths = planned.feature_lengths
                if self.config.use_gpu:
                    features = features.cuda()
                    lengths = lengths.cuda()
                    if self.config.use_half:
                        features = features.half()
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
            inference_elapsed += time.time() - start_time
            with self._measure_stage("result_formatting"):
                for item, hypotheses_for_item in zip(
                    planned.items, hypotheses
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
    def process(self, batch_uttid, batch_wav_path):
        batch_uttid_origin = batch_uttid
        try:
            with self._measure_stage("fbank"):
                items = self.feat_extractor.extract_many(
                    batch_wav_path,
                    batch_uttid,
                    max_audio_seconds=self.config.max_audio_seconds,
                )
            if not items:
                return [
                    {"uttid": uttid, "lang": ""}
                    for uttid in batch_uttid_origin
                ]
        except:
            traceback.print_exc()
            return [
                {"uttid": uttid, "lang": ""}
                for uttid in batch_uttid_origin
            ]

        try:
            return self._infer_items(items)
        except Exception:
            traceback.print_exc()
            return []


def load_fireredlid_model(model_path):
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    #print(package["args"])
    model = FireRedLidAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    return model
