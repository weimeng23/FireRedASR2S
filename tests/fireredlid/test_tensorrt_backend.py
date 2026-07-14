import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

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
