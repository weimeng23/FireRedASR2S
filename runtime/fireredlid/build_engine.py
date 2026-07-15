#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path

import onnx
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fireredasr2s.fireredlid.runtime.tensorrt_backend import (
    BackendUnavailableError,
    sha256_file,
)


INPUT_NAMES = ["features", "feature_lengths"]
OUTPUT_NAMES = [
    "encoder_outputs",
    "encoder_lengths",
    "encoder_mask",
]


def load_profile(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("unsupported TensorRT profile schema")
    profiles = data.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("TensorRT profiles must not be empty")
    for profile in profiles:
        features = profile.get("features", {})
        lengths = profile.get("feature_lengths", {})
        for point in ("min", "opt", "max"):
            feature_shape = features.get(point)
            length_shape = lengths.get(point)
            if (
                not isinstance(feature_shape, list)
                or len(feature_shape) != 3
                or not all(
                    isinstance(value, int) and value > 0
                    for value in feature_shape
                )
            ):
                raise ValueError(
                    f"invalid features {point} shape in profile"
                )
            if (
                not isinstance(length_shape, list)
                or len(length_shape) != 1
                or length_shape[0] != feature_shape[0]
            ):
                raise ValueError(
                    f"feature_lengths {point} batch must match features"
                )
        for axis in range(3):
            if not (
                features["min"][axis]
                <= features["opt"][axis]
                <= features["max"][axis]
            ):
                raise ValueError("invalid TensorRT profile bounds")
        if features["min"][2] != 80 or features["max"][2] != 80:
            raise ValueError("feature dimension must be 80")
    return data


def flatten_profiles(profile_config):
    return [
        {
            "name": profile["name"],
            "min": list(profile["features"]["min"]),
            "opt": list(profile["features"]["opt"]),
            "max": list(profile["features"]["max"]),
        }
        for profile in profile_config["profiles"]
    ]


def _update_digest_from_file(digest, label, path):
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)


def sha256_onnx_bundle(onnx_path):
    onnx_path = Path(onnx_path)
    model = onnx.load(onnx_path, load_external_data=False)
    external_locations = set()
    for initializer in model.graph.initializer:
        for item in initializer.external_data:
            if item.key == "location":
                external_locations.add(item.value)
    digest = hashlib.sha256()
    _update_digest_from_file(digest, onnx_path.name, onnx_path)
    for location in sorted(external_locations):
        _update_digest_from_file(
            digest,
            location,
            onnx_path.parent / location,
        )
    return digest.hexdigest()


def inspect_onnx_bundle(onnx_path):
    onnx_path = Path(onnx_path).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"missing ONNX graph: {onnx_path}")
    model = onnx.load(onnx_path, load_external_data=False)
    opsets = [
        item.version for item in model.opset_import if item.domain == ""
    ]
    if opsets != [17]:
        raise ValueError(f"expected ONNX opset 17, got {opsets}")
    input_names = [value.name for value in model.graph.input]
    output_names = [value.name for value in model.graph.output]
    if input_names != INPUT_NAMES or output_names != OUTPUT_NAMES:
        raise ValueError(
            f"ONNX tensor contract mismatch: inputs={input_names}, "
            f"outputs={output_names}"
        )
    locations = sorted(
        {
            item.value
            for initializer in model.graph.initializer
            for item in initializer.external_data
            if item.key == "location"
        }
    )
    for location in locations:
        relative_path = Path(location)
        if relative_path.is_absolute():
            raise ValueError(
                f"absolute external ONNX data location is not allowed: "
                f"{location}"
            )
        resolved_path = (onnx_path.parent / relative_path).resolve()
        try:
            resolved_path.relative_to(onnx_path.parent)
        except ValueError as error:
            raise ValueError(
                f"external ONNX data location escapes graph directory: "
                f"{location}"
            ) from error
    missing = [
        location
        for location in locations
        if not (onnx_path.parent / location).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing external ONNX data: {missing}")
    onnx.checker.check_model(onnx_path)
    return {
        "opset": 17,
        "input_names": input_names,
        "output_names": output_names,
        "external_file_count": len(locations),
        "missing_external_files": missing,
        "onnx_bundle_sha256": sha256_onnx_bundle(onnx_path),
    }


def preflight_artifacts(onnx_path, checkpoint_path, profile_path):
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    profile_path = Path(profile_path).resolve()
    load_profile(profile_path)
    report = inspect_onnx_bundle(onnx_path)
    report.update(
        {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "profile_sha256": sha256_file(profile_path),
            "ready_for_tensorrt_build": True,
        }
    )
    return report


def require_tensorrt():
    if (
        sys.platform != "linux"
        or not torch.cuda.is_available()
        or importlib.util.find_spec("tensorrt") is None
    ):
        raise BackendUnavailableError(
            "TensorRT engine build requires Linux NVIDIA with TensorRT installed"
        )
    import tensorrt as trt

    if not hasattr(trt, "int64") or not hasattr(trt, "uint8"):
        raise BackendUnavailableError(
            "TensorRT build requires INT64 and UINT8 tensor support"
        )
    return trt


def _network_tensors(network, count_name, getter_name):
    count = getattr(network, count_name)
    getter = getattr(network, getter_name)
    return {getter(index).name: getter(index) for index in range(count)}


def write_manifest(
    output_dir,
    profile_config,
    checkpoint_path,
    onnx_path,
    trt_version,
):
    output_dir = Path(output_dir)
    manifest = {
        "schema_version": 1,
        "precision": "float16",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx_sha256": sha256_onnx_bundle(onnx_path),
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "tensor_dtypes": {
            "features": "float16",
            "feature_lengths": "int64",
            "encoder_outputs": "float16",
            "encoder_lengths": "int64",
            "encoder_mask": "uint8",
        },
        "profiles": flatten_profiles(profile_config),
        "environment": {
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "tensorrt": trt_version,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def build_engine(onnx_path, output_dir, profile_path, checkpoint_path):
    preflight_artifacts(onnx_path, checkpoint_path, profile_path)
    trt = require_tensorrt()
    onnx_path = Path(onnx_path).resolve()
    output_dir = Path(output_dir)
    profile_path = Path(profile_path)
    profile_config = load_profile(profile_path)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not hasattr(parser, "parse_from_file"):
        raise BackendUnavailableError(
            "TensorRT OnnxParser.parse_from_file is required for external ONNX data"
        )
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(
            str(parser.get_error(index))
            for index in range(parser.num_errors)
        )
        raise RuntimeError(errors)

    inputs = _network_tensors(network, "num_inputs", "get_input")
    outputs = _network_tensors(network, "num_outputs", "get_output")
    if list(inputs) != INPUT_NAMES or list(outputs) != OUTPUT_NAMES:
        raise RuntimeError(
            f"ONNX tensor contract mismatch: inputs={list(inputs)}, "
            f"outputs={list(outputs)}"
        )
    inputs["features"].dtype = trt.float16
    inputs["feature_lengths"].dtype = trt.int64
    outputs["encoder_outputs"].dtype = trt.float16
    outputs["encoder_lengths"].dtype = trt.int64
    outputs["encoder_mask"].dtype = trt.uint8

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    for item in profile_config["profiles"]:
        profile = builder.create_optimization_profile()
        for name in INPUT_NAMES:
            shapes = item[name]
            if not profile.set_shape(
                name,
                tuple(shapes["min"]),
                tuple(shapes["opt"]),
                tuple(shapes["max"]),
            ):
                raise RuntimeError(
                    f"TensorRT rejected profile {item['name']} input {name}"
                )
        config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT returned an empty serialized engine")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "encoder.plan").write_bytes(bytes(serialized))
    destination_profile = output_dir / "profiles.yaml"
    if profile_path.resolve() != destination_profile.resolve():
        shutil.copy2(profile_path, destination_profile)
    write_manifest(
        output_dir,
        profile_config,
        checkpoint_path,
        onnx_path,
        trt.__version__,
    )
    return output_dir / "encoder.plan"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a FireRedLID FP16 TensorRT Encoder engine."
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report")
    return parser.parse_args()


def main():
    args = parse_args()
    report = preflight_artifacts(
        args.onnx,
        args.checkpoint,
        args.profiles,
    )
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return
    path = build_engine(
        args.onnx,
        args.output_dir,
        args.profiles,
        args.checkpoint,
    )
    print(path)


if __name__ == "__main__":
    main()
