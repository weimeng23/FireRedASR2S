import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from .encoder_backend import EncoderBackend, EncoderResult


class ArtifactMismatchError(RuntimeError):
    pass


class BackendUnavailableError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EngineManifest:
    precision: str
    checkpoint_sha256: str
    onnx_sha256: str
    engine_sha256: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    tensor_dtypes: dict[str, str]
    profiles: tuple[dict, ...]
    environment: dict[str, str]

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 2:
            raise ArtifactMismatchError(
                "unsupported engine manifest schema"
            )
        expected_inputs = ["features", "feature_lengths"]
        expected_outputs = [
            "encoder_outputs",
            "encoder_lengths",
            "encoder_mask",
        ]
        if data.get("input_names") != expected_inputs:
            raise ArtifactMismatchError("engine input contract mismatch")
        if data.get("output_names") != expected_outputs:
            raise ArtifactMismatchError("engine output contract mismatch")
        if data.get("precision") != "float16":
            raise ArtifactMismatchError("engine precision must be float16")
        for name in (
            "checkpoint_sha256",
            "onnx_sha256",
            "engine_sha256",
        ):
            value = data.get(name)
            if not isinstance(value, str) or len(value) != 64:
                raise ArtifactMismatchError(f"invalid {name}")
        expected_dtypes = {
            "features": "float16",
            "feature_lengths": "int64",
            "encoder_outputs": "float16",
            "encoder_lengths": "int64",
            "encoder_mask": "uint8",
        }
        if data.get("tensor_dtypes") != expected_dtypes:
            raise ArtifactMismatchError(
                "engine tensor dtype contract mismatch"
            )
        profiles = data.get("profiles")
        if not isinstance(profiles, list) or not profiles:
            raise ArtifactMismatchError("engine profiles must not be empty")
        for profile in profiles:
            if (
                not isinstance(profile, dict)
                or not isinstance(profile.get("name"), str)
                or not profile["name"].strip()
            ):
                raise ArtifactMismatchError("invalid engine profile")
            minimum = profile.get("min")
            optimum = profile.get("opt")
            maximum = profile.get("max")
            if not all(
                isinstance(shape, list)
                and len(shape) == 3
                and all(
                    isinstance(dimension, int) and dimension > 0
                    for dimension in shape
                )
                for shape in (minimum, optimum, maximum)
            ):
                raise ArtifactMismatchError("invalid engine profile shape")
            if not all(
                minimum[index] <= optimum[index] <= maximum[index]
                for index in range(3)
            ):
                raise ArtifactMismatchError("invalid engine profile bounds")
            if minimum[2] != 80 or maximum[2] != 80:
                raise ArtifactMismatchError(
                    "engine profile feature dimension must be 80"
                )
        environment = data.get("environment")
        required_environment = (
            "gpu",
            "compute_capability",
            "python",
            "pytorch",
            "cuda",
            "tensorrt",
        )
        if not isinstance(environment, dict) or any(
            not isinstance(environment.get(name), str)
            or not environment[name].strip()
            for name in required_environment
        ):
            raise ArtifactMismatchError("invalid engine environment metadata")
        return cls(
            precision=data["precision"],
            checkpoint_sha256=data["checkpoint_sha256"],
            onnx_sha256=data["onnx_sha256"],
            engine_sha256=data["engine_sha256"],
            input_names=tuple(data["input_names"]),
            output_names=tuple(data["output_names"]),
            tensor_dtypes=dict(data["tensor_dtypes"]),
            profiles=tuple(profiles),
            environment=dict(environment),
        )

    @property
    def max_batch(self):
        return max(profile["max"][0] for profile in self.profiles)

    def select_profile(self, shape):
        if len(shape) != 3:
            raise ArtifactMismatchError(
                f"expected feature shape [B, T, 80], got {shape}"
            )
        for index, profile in enumerate(self.profiles):
            minimum = profile["min"]
            maximum = profile["max"]
            if all(
                minimum[axis] <= shape[axis] <= maximum[axis]
                for axis in range(3)
            ):
                return index
        raise ArtifactMismatchError(
            f"feature shape {shape} is outside engine profiles"
        )

    def validate_checkpoint(self, checkpoint_path):
        actual = sha256_file(checkpoint_path)
        if actual != self.checkpoint_sha256:
            raise ArtifactMismatchError(
                "checkpoint SHA-256 differs from engine manifest"
            )

    def validate_engine(self, engine_path):
        actual = sha256_file(engine_path)
        if actual != self.engine_sha256:
            raise ArtifactMismatchError(
                "engine SHA-256 differs from engine manifest"
            )


class TensorRTEncoderBackend(EncoderBackend):
    def __init__(self, engine_dir, checkpoint_path=None):
        super().__init__()
        self.engine_dir = Path(engine_dir)
        self.manifest = EngineManifest.load(
            self.engine_dir / "manifest.json"
        )
        if checkpoint_path is not None:
            self.manifest.validate_checkpoint(checkpoint_path)
        if (
            sys.platform != "linux"
            or not torch.cuda.is_available()
            or importlib.util.find_spec("tensorrt") is None
        ):
            raise BackendUnavailableError(
                "TensorRT backend requires Linux NVIDIA with TensorRT installed"
            )

        import tensorrt as trt

        self._trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        engine_path = self.engine_dir / "encoder.plan"
        if not engine_path.is_file():
            raise ArtifactMismatchError(f"missing TensorRT engine: {engine_path}")
        self.manifest.validate_engine(engine_path)
        self.engine = self.runtime.deserialize_cuda_engine(
            engine_path.read_bytes()
        )
        if self.engine is None:
            raise ArtifactMismatchError(
                "failed to deserialize TensorRT encoder engine"
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise ArtifactMismatchError(
                "failed to create TensorRT execution context"
            )
        self._validate_engine_contract()
        self._active_profile = 0

    @property
    def max_batch(self):
        return self.manifest.max_batch

    def _validate_engine_contract(self):
        names = tuple(
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        )
        expected_names = self.manifest.input_names + self.manifest.output_names
        if set(names) != set(expected_names):
            raise ArtifactMismatchError(
                f"engine tensor names differ from manifest: {names}"
            )
        trt = self._trt
        expected_dtypes = {
            "features": trt.float16,
            "feature_lengths": trt.int64,
            "encoder_outputs": trt.float16,
            "encoder_lengths": trt.int64,
            "encoder_mask": trt.uint8,
        }
        for name, expected_dtype in expected_dtypes.items():
            actual_dtype = self.engine.get_tensor_dtype(name)
            if actual_dtype != expected_dtype:
                raise ArtifactMismatchError(
                    f"engine tensor {name} has dtype {actual_dtype}, "
                    f"expected {expected_dtype}"
                )

    def encode(self, features, feature_lengths):
        shape = tuple(features.shape)
        profile_index = self.manifest.select_profile(shape)
        features = features.to(
            device="cuda",
            dtype=torch.float16,
        ).contiguous()
        feature_lengths = feature_lengths.to(
            device="cuda",
            dtype=torch.int64,
        ).contiguous()
        stream = torch.cuda.current_stream()
        if profile_index != self._active_profile:
            if not self.context.set_optimization_profile_async(
                profile_index,
                stream.cuda_stream,
            ):
                raise RuntimeError(
                    f"failed to select TensorRT profile {profile_index}"
                )
            self._active_profile = profile_index
        if self.context.set_input_shape("features", shape) is False:
            raise RuntimeError(f"TensorRT rejected feature shape {shape}")
        lengths_shape = tuple(feature_lengths.shape)
        if (
            self.context.set_input_shape(
                "feature_lengths", lengths_shape
            )
            is False
        ):
            raise RuntimeError(
                f"TensorRT rejected lengths shape {lengths_shape}"
            )
        output_shape = tuple(
            self.context.get_tensor_shape("encoder_outputs")
        )
        output_lengths_shape = tuple(
            self.context.get_tensor_shape("encoder_lengths")
        )
        mask_shape = tuple(
            self.context.get_tensor_shape("encoder_mask")
        )
        if any(
            dimension < 0
            for output in (
                output_shape,
                output_lengths_shape,
                mask_shape,
            )
            for dimension in output
        ):
            raise RuntimeError("TensorRT output shape remains dynamic")
        outputs = torch.empty(
            output_shape,
            device="cuda",
            dtype=torch.float16,
        )
        lengths = torch.empty(
            output_lengths_shape,
            device="cuda",
            dtype=torch.int64,
        )
        mask = torch.empty(
            mask_shape,
            device="cuda",
            dtype=torch.uint8,
        )
        tensors = {
            "features": features,
            "feature_lengths": feature_lengths,
            "encoder_outputs": outputs,
            "encoder_lengths": lengths,
            "encoder_mask": mask,
        }
        for name, tensor in tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT encoder execution failed")
        return EncoderResult(
            outputs=outputs,
            lengths=lengths,
            mask=mask,
        )
