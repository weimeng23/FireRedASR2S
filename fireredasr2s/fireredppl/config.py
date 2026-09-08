"""Validated service and inference budgets (all lengths count padded work)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BucketPolicy(Config):
    max_seconds: float = Field(gt=0, allow_inf_nan=False)
    max_batch_size: int = Field(gt=0)


class ServerConfig(Config):
    host: str = "0.0.0.0"
    port: int = Field(default=12345, ge=1, le=65535)
    decode_workers: int = Field(default=16, gt=0)
    queue_capacity: int = Field(default=512, gt=0)
    queue_max_candidates: int = Field(default=2048, gt=0)
    queue_max_tokens: int = Field(default=131072, gt=0)
    max_request_bytes: int = Field(default=16 * 1024 * 1024, gt=0)
    request_timeout_s: float = Field(default=30.0, gt=0, allow_inf_nan=False)


class ModelConfig(Config):
    model_dir: str = ""
    scorer_type: Literal["aed"] = "aed"
    max_audio_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    max_candidates: int = Field(default=64, gt=0)
    max_text_chars: int = Field(default=4096, gt=0)
    max_text_tokens: int = Field(default=512, gt=0)
    include_eos: bool = True
    softmax_smoothing: float = Field(default=1.0, gt=0, allow_inf_nan=False)


class RuntimeConfig(Config):
    use_gpu: bool = True
    encoder_precision: Literal["fp32", "fp16", "bf16"] = "fp16"
    decoder_precision: Literal["fp32", "fp16", "bf16"] = "fp16"


class SchedulerConfig(Config):
    max_batch_delay_ms: float = Field(default=10.0, ge=0, allow_inf_nan=False)
    buckets: list[BucketPolicy] = Field(default_factory=lambda: [
        BucketPolicy(max_seconds=5.0, max_batch_size=32),
        BucketPolicy(max_seconds=15.0, max_batch_size=16),
        BucketPolicy(max_seconds=30.0, max_batch_size=8),
        BucketPolicy(max_seconds=60.0, max_batch_size=2),
    ])
    encoder_max_padded_frames: int = Field(default=24000, gt=0)
    encoder_max_attention_elements: int = Field(default=8000000, gt=0)
    decoder_max_batch_size: int = Field(default=32, gt=0)
    decoder_max_padded_tokens: int = Field(default=4096, gt=0)
    decoder_max_attention_elements: int = Field(default=8000000, gt=0)

    @model_validator(mode="after")
    def validate_buckets(self):
        bounds = [bucket.max_seconds for bucket in self.buckets]
        if not bounds or bounds != sorted(set(bounds)):
            raise ValueError("bucket boundaries must be nonempty and strictly increasing")
        return self


class Settings(Config):
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    @model_validator(mode="after")
    def validate_limits(self):
        if self.scheduler.buckets[-1].max_seconds < self.model.max_audio_seconds:
            raise ValueError("buckets must cover max_audio_seconds")
        if not self.runtime.use_gpu and (
            self.runtime.encoder_precision != "fp32" or self.runtime.decoder_precision != "fp32"
        ):
            raise ValueError("CPU inference requires fp32 precision")
        return self
