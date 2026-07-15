import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, external_data_helper, helper, numpy_helper

from fireredasr2s.fireredlid.runtime.tensorrt_backend import (
    ArtifactMismatchError,
    BackendUnavailableError,
    EngineManifest,
    TensorRTEncoderBackend,
)


BUILD_SCRIPT = Path("runtime/fireredlid/build_engine.py")


def load_build_module():
    spec = importlib.util.spec_from_file_location(
        "fireredlid_build_engine",
        BUILD_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_data(checkpoint_sha256=None):
    return {
        "schema_version": 1,
        "precision": "float16",
        "checkpoint_sha256": checkpoint_sha256 or "0" * 64,
        "onnx_sha256": "1" * 64,
        "input_names": ["features", "feature_lengths"],
        "output_names": [
            "encoder_outputs",
            "encoder_lengths",
            "encoder_mask",
        ],
        "profiles": [
            {
                "name": "default",
                "min": [1, 1, 80],
                "opt": [1, 1000, 80],
                "max": [4, 6000, 80],
            }
        ],
    }


def write_manifest(path, data=None):
    path.write_text(
        json.dumps(data or manifest_data()),
        encoding="utf-8",
    )


def write_valid_preflight_bundle(tmp_path):
    onnx_path = tmp_path / "encoder.onnx"
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    weight = numpy_helper.from_array(
        np.eye(80, dtype=np.float32),
        name="weight",
    )
    external_data_helper.set_external_data(
        weight,
        location="data/weight",
    )
    reduce_axes = numpy_helper.from_array(
        np.array([2], dtype=np.int64),
        name="reduce_axes",
    )
    channel_axis = numpy_helper.from_array(
        np.array([1], dtype=np.int64),
        name="channel_axis",
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMul",
                ["features", "weight"],
                ["encoder_outputs"],
            ),
            helper.make_node(
                "Identity",
                ["feature_lengths"],
                ["encoder_lengths"],
            ),
            helper.make_node(
                "ReduceSum",
                ["features", "reduce_axes"],
                ["mask_values"],
                keepdims=0,
            ),
            helper.make_node(
                "Cast",
                ["mask_values"],
                ["mask_uint8"],
                to=TensorProto.UINT8,
            ),
            helper.make_node(
                "Unsqueeze",
                ["mask_uint8", "channel_axis"],
                ["encoder_mask"],
            ),
        ],
        "preflight-test",
        [
            helper.make_tensor_value_info(
                "features",
                TensorProto.FLOAT,
                ["batch", "time", 80],
            ),
            helper.make_tensor_value_info(
                "feature_lengths",
                TensorProto.INT64,
                ["batch"],
            ),
        ],
        [
            helper.make_tensor_value_info(
                "encoder_outputs",
                TensorProto.FLOAT,
                ["batch", "time", 80],
            ),
            helper.make_tensor_value_info(
                "encoder_lengths",
                TensorProto.INT64,
                ["batch"],
            ),
            helper.make_tensor_value_info(
                "encoder_mask",
                TensorProto.UINT8,
                ["batch", 1, "time"],
            ),
        ],
        [weight, reduce_axes, channel_axis],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.save_model(model, onnx_path)

    checkpoint = tmp_path / "model.pth.tar"
    checkpoint.write_bytes(b"checkpoint")
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(
        """
schema_version: 1
profiles:
  - name: default
    features:
      min: [1, 1, 80]
      opt: [2, 1000, 80]
      max: [4, 6000, 80]
    feature_lengths:
      min: [1]
      opt: [2]
      max: [4]
""".strip(),
        encoding="utf-8",
    )
    return onnx_path, checkpoint, profiles


def test_manifest_exposes_max_batch_and_selects_profile(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path)

    manifest = EngineManifest.load(manifest_path)

    assert manifest.max_batch == 4
    assert manifest.select_profile((4, 6000, 80)) == 0
    with pytest.raises(ArtifactMismatchError, match="outside engine profiles"):
        manifest.select_profile((5, 1000, 80))


def test_tensorrt_backend_fails_cleanly_without_tensorrt(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "encoder.plan").write_bytes(b"not-an-engine")
    write_manifest(tmp_path / "manifest.json")
    monkeypatch.setattr(sys, "platform", "darwin")

    with pytest.raises(BackendUnavailableError, match="Linux NVIDIA"):
        TensorRTEncoderBackend(tmp_path)


def test_manifest_rejects_checkpoint_mismatch(tmp_path):
    checkpoint = tmp_path / "model.pth.tar"
    checkpoint.write_bytes(b"checkpoint-a")
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        manifest_data(
            checkpoint_sha256=hashlib.sha256(
                b"checkpoint-b"
            ).hexdigest()
        ),
    )
    manifest = EngineManifest.load(manifest_path)

    with pytest.raises(ArtifactMismatchError, match="checkpoint SHA-256"):
        manifest.validate_checkpoint(checkpoint)


def test_build_profile_loader_flattens_feature_shapes(tmp_path):
    module = load_build_module()
    profile_path = tmp_path / "profiles.yaml"
    profile_path.write_text(
        """
schema_version: 1
profiles:
  - name: default
    features:
      min: [1, 1, 80]
      opt: [2, 1000, 80]
      max: [4, 6000, 80]
    feature_lengths:
      min: [1]
      opt: [2]
      max: [4]
""".strip(),
        encoding="utf-8",
    )

    config = module.load_profile(profile_path)

    assert module.flatten_profiles(config) == [
        {
            "name": "default",
            "min": [1, 1, 80],
            "opt": [2, 1000, 80],
            "max": [4, 6000, 80],
        }
    ]


def test_preflight_reports_exact_encoder_contract(tmp_path):
    module = load_build_module()
    onnx_path, checkpoint, profiles = write_valid_preflight_bundle(tmp_path)

    report = module.preflight_artifacts(onnx_path, checkpoint, profiles)

    assert report["opset"] == 17
    assert report["input_names"] == ["features", "feature_lengths"]
    assert report["output_names"] == [
        "encoder_outputs",
        "encoder_lengths",
        "encoder_mask",
    ]
    assert report["external_file_count"] > 0
    assert report["missing_external_files"] == []
    assert report["ready_for_tensorrt_build"] is True


def test_preflight_rejects_missing_external_weight(tmp_path):
    module = load_build_module()
    onnx_path, checkpoint, profiles = write_valid_preflight_bundle(tmp_path)
    next((onnx_path.parent / "data").iterdir()).unlink()

    with pytest.raises(FileNotFoundError, match="external ONNX data"):
        module.preflight_artifacts(onnx_path, checkpoint, profiles)


def test_preflight_runs_before_tensorrt_platform_check(tmp_path, monkeypatch):
    module = load_build_module()
    onnx_path, checkpoint, profiles = write_valid_preflight_bundle(tmp_path)
    monkeypatch.setattr(
        module,
        "require_tensorrt",
        lambda: (_ for _ in ()).throw(
            AssertionError("TensorRT import must not run during preflight")
        ),
    )

    report = module.preflight_artifacts(onnx_path, checkpoint, profiles)

    assert report["ready_for_tensorrt_build"] is True
