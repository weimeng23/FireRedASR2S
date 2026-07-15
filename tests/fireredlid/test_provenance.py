import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from fireredasr2s.fireredlid.runtime.provenance import (
    collect_git_info,
    collect_provenance,
    file_artifact,
    onnx_bundle_artifact,
)


def _write_external_onnx(tmp_path):
    graph_path = tmp_path / "encoder.onnx"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    weight = numpy_helper.from_array(
        np.eye(2, dtype=np.float32),
        name="weight",
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["features", "weight"], ["outputs"])],
        "provenance-test",
        [
            helper.make_tensor_value_info(
                "features",
                TensorProto.FLOAT,
                [None, 2],
            )
        ],
        [
            helper.make_tensor_value_info(
                "outputs",
                TensorProto.FLOAT,
                [None, 2],
            )
        ],
        [weight],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.save_model(
        model,
        graph_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="data/weights.bin",
        size_threshold=0,
    )
    return graph_path, data_dir / "weights.bin"


def test_collect_provenance_has_exact_common_envelope_and_is_json_serializable(
    tmp_path,
):
    artifact_path = tmp_path / "input.txt"
    artifact_path.write_bytes(b"acceptance input")

    report = collect_provenance(
        arguments={
            "path": artifact_path,
            "seconds": (1, 5),
            "enabled": True,
            "optional": None,
        },
        input_artifacts={"input": file_artifact(artifact_path)},
        repo_root=tmp_path,
    )

    assert set(report) == {
        "schema_version",
        "git",
        "platform",
        "versions",
        "cuda",
        "input_artifacts",
        "arguments",
    }
    assert set(report["git"]) == {"branch", "commit"}
    assert set(report["platform"]) == {
        "platform",
        "system",
        "release",
        "machine",
    }
    assert set(report["versions"]) == {
        "python",
        "pytorch",
        "cuda",
        "tensorrt",
        "onnx",
        "onnxruntime",
    }
    assert set(report["cuda"]) == {
        "cuda_visible_devices",
        "available",
        "device_count",
        "current_device",
        "gpu_name",
        "compute_capability",
        "devices",
        "nvidia_smi",
    }
    assert set(report["cuda"]["nvidia_smi"]) == {
        "available",
        "command",
        "return_code",
        "devices",
    }
    assert report["git"] == {"branch": None, "commit": None}
    assert report["arguments"] == {
        "path": str(artifact_path),
        "seconds": [1, 5],
        "enabled": True,
        "optional": None,
    }
    assert report["input_artifacts"]["input"]["sha256"] == hashlib.sha256(
        b"acceptance input"
    ).hexdigest()
    json.dumps(report)


def test_collect_git_info_is_deterministic_outside_git(tmp_path):
    assert collect_git_info(tmp_path) == {"branch": None, "commit": None}


def test_onnx_bundle_artifact_hashes_graph_and_all_external_data(tmp_path):
    graph_path, weights_path = _write_external_onnx(tmp_path)
    artifact = onnx_bundle_artifact(graph_path)

    expected_bundle = hashlib.sha256()
    for label, path in (
        ("encoder.onnx", graph_path),
        ("data/weights.bin", weights_path),
    ):
        expected_bundle.update(label.encode("utf-8"))
        expected_bundle.update(b"\0")
        expected_bundle.update(path.read_bytes())

    assert artifact == {
        "path": str(graph_path.resolve()),
        "sha256": expected_bundle.hexdigest(),
        "file_count": 2,
        "external_file_count": 1,
        "files": [
            {
                "path": "encoder.onnx",
                "sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
            },
            {
                "path": "data/weights.bin",
                "sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            },
        ],
    }
