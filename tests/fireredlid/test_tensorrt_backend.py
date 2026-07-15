import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, external_data_helper, helper, numpy_helper

import fireredasr2s.fireredlid.runtime.tensorrt_backend as tensorrt_backend
from fireredasr2s.fireredlid.runtime.tensorrt_backend import (
    ArtifactMismatchError,
    BackendUnavailableError,
    EngineManifest,
    TensorRTEncoderBackend,
    _bind_and_execute,
    _configure_context,
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


def manifest_data(
    checkpoint_sha256=None,
    *,
    schema_version=2,
    engine_sha256=None,
):
    return {
        "schema_version": schema_version,
        "precision": "float16",
        "checkpoint_sha256": checkpoint_sha256 or "0" * 64,
        "onnx_sha256": "1" * 64,
        "engine_sha256": engine_sha256 or "2" * 64,
        "input_names": ["features", "feature_lengths"],
        "output_names": [
            "encoder_outputs",
            "encoder_lengths",
            "encoder_mask",
        ],
        "tensor_dtypes": {
            "features": "float16",
            "feature_lengths": "int64",
            "encoder_outputs": "float16",
            "encoder_lengths": "int64",
            "encoder_mask": "uint8",
        },
        "profiles": [
            {
                "name": "default",
                "min": [1, 1, 80],
                "opt": [1, 1000, 80],
                "max": [4, 6000, 80],
            }
        ],
        "environment": {
            "gpu": "Test GPU",
            "compute_capability": "8.0",
            "python": "3.11.8",
            "pytorch": "2.10.0",
            "cuda": "12.9",
            "tensorrt": "10.14.1",
        },
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


class FakeContext:
    def __init__(
        self,
        *,
        rejected_shape=None,
        output_shapes=None,
        profile_switch_success=True,
        rejected_address=None,
        execute_success=True,
    ):
        self.rejected_shape = rejected_shape
        self.output_shapes = output_shapes or {
            "encoder_outputs": (2, 250, 512),
            "encoder_lengths": (2,),
            "encoder_mask": (2, 1, 250),
        }
        self.profile_switch_success = profile_switch_success
        self.rejected_address = rejected_address
        self.execute_success = execute_success
        self.profile_calls = []
        self.shape_calls = []
        self.address_calls = []
        self.execute_calls = []

    def set_optimization_profile_async(self, profile_index, stream_handle):
        self.profile_calls.append((profile_index, stream_handle))
        return self.profile_switch_success

    def set_input_shape(self, name, shape):
        self.shape_calls.append((name, shape))
        return name != self.rejected_shape

    def get_tensor_shape(self, name):
        return self.output_shapes[name]

    def set_tensor_address(self, name, address):
        self.address_calls.append((name, address))
        return name != self.rejected_address

    def execute_async_v3(self, stream_handle):
        self.execute_calls.append(stream_handle)
        return self.execute_success


class FakeTensor:
    def __init__(self, address):
        self.address = address

    def data_ptr(self):
        return self.address


def fake_tensors_with_addresses():
    return {
        "features": FakeTensor(1),
        "feature_lengths": FakeTensor(2),
        "encoder_outputs": FakeTensor(3),
        "encoder_lengths": FakeTensor(4),
        "encoder_mask": FakeTensor(5),
    }


def test_context_selects_profile_zero_explicitly():
    context = FakeContext()

    active, shapes = _configure_context(
        context,
        profile_index=0,
        active_profile=-1,
        feature_shape=(2, 1000, 80),
        lengths_shape=(2,),
        stream_handle=123,
    )

    assert context.profile_calls == [(0, 123)]
    assert active == 0
    assert shapes["encoder_outputs"] == (2, 250, 512)


def test_context_keeps_the_active_profile_without_switching():
    context = FakeContext()

    active, _ = _configure_context(
        context,
        profile_index=0,
        active_profile=0,
        feature_shape=(2, 1000, 80),
        lengths_shape=(2,),
        stream_handle=123,
    )

    assert context.profile_calls == []
    assert active == 0


def test_context_reports_profile_switch_failure():
    context = FakeContext(profile_switch_success=False)

    with pytest.raises(RuntimeError, match="TensorRT profile 1"):
        _configure_context(
            context,
            profile_index=1,
            active_profile=0,
            feature_shape=(2, 1000, 80),
            lengths_shape=(2,),
            stream_handle=123,
        )

    assert context.shape_calls == []


@pytest.mark.parametrize(
    ("rejected_shape", "message"),
    [
        ("features", "feature shape"),
        ("feature_lengths", "lengths shape"),
    ],
)
def test_context_reports_rejected_input_shape(rejected_shape, message):
    context = FakeContext(rejected_shape=rejected_shape)

    with pytest.raises(RuntimeError, match=message):
        _configure_context(
            context,
            profile_index=0,
            active_profile=0,
            feature_shape=(2, 1000, 80),
            lengths_shape=(2,),
            stream_handle=123,
        )


@pytest.mark.parametrize(
    "output_name",
    ["encoder_outputs", "encoder_lengths", "encoder_mask"],
)
def test_context_rejects_unresolved_output_shape(output_name):
    output_shapes = {
        "encoder_outputs": (2, 250, 512),
        "encoder_lengths": (2,),
        "encoder_mask": (2, 1, 250),
    }
    output_shapes[output_name] = (-1,)
    context = FakeContext(output_shapes=output_shapes)

    with pytest.raises(RuntimeError, match="output shape remains dynamic"):
        _configure_context(
            context,
            profile_index=0,
            active_profile=0,
            feature_shape=(2, 1000, 80),
            lengths_shape=(2,),
            stream_handle=123,
        )


def test_binding_addresses_and_executes():
    context = FakeContext()
    tensors = fake_tensors_with_addresses()

    assert _bind_and_execute(context, tensors, stream_handle=123) is None

    assert context.address_calls == [
        (name, tensor.data_ptr()) for name, tensor in tensors.items()
    ]
    assert context.execute_calls == [123]


def test_binding_failure_names_the_tensor():
    context = FakeContext(rejected_address="encoder_mask")
    tensors = fake_tensors_with_addresses()

    with pytest.raises(RuntimeError, match="encoder_mask"):
        _bind_and_execute(context, tensors, stream_handle=123)

    assert context.execute_calls == []


def test_execution_failure_is_reported():
    context = FakeContext(execute_success=False)

    with pytest.raises(RuntimeError, match="encoder execution failed"):
        _bind_and_execute(
            context,
            fake_tensors_with_addresses(),
            stream_handle=123,
        )


def test_manifest_exposes_max_batch_and_selects_profile(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path)

    manifest = EngineManifest.load(manifest_path)

    assert manifest.max_batch == 4
    assert manifest.engine_sha256 == "2" * 64
    assert manifest.tensor_dtypes["encoder_mask"] == "uint8"
    assert manifest.environment["compute_capability"] == "8.0"
    assert manifest.select_profile((4, 6000, 80)) == 0
    with pytest.raises(ArtifactMismatchError, match="outside engine profiles"):
        manifest.select_profile((5, 1000, 80))


def test_manifest_rejects_schema_v1(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        manifest_data(schema_version=1),
    )

    with pytest.raises(ArtifactMismatchError, match="manifest schema"):
        EngineManifest.load(manifest_path)


def test_manifest_rejects_engine_hash_mismatch(tmp_path):
    engine = tmp_path / "encoder.plan"
    engine.write_bytes(b"engine-a")
    data = manifest_data(
        engine_sha256=hashlib.sha256(b"engine-b").hexdigest(),
    )
    write_manifest(tmp_path / "manifest.json", data)

    manifest = EngineManifest.load(tmp_path / "manifest.json")
    with pytest.raises(ArtifactMismatchError, match="engine SHA-256"):
        manifest.validate_engine(engine)


def test_manifest_requires_declared_tensor_dtypes(tmp_path):
    data = manifest_data()
    data["tensor_dtypes"].pop("encoder_mask")
    write_manifest(tmp_path / "manifest.json", data)

    with pytest.raises(ArtifactMismatchError, match="tensor dtype contract"):
        EngineManifest.load(tmp_path / "manifest.json")


def test_manifest_rejects_undeclared_tensor_dtype(tmp_path):
    data = manifest_data()
    data["tensor_dtypes"]["extra"] = "float32"
    write_manifest(tmp_path / "manifest.json", data)

    with pytest.raises(ArtifactMismatchError, match="tensor dtype contract"):
        EngineManifest.load(tmp_path / "manifest.json")


@pytest.mark.parametrize(
    "field",
    ["checkpoint_sha256", "onnx_sha256", "engine_sha256"],
)
def test_manifest_requires_64_character_hashes(tmp_path, field):
    data = manifest_data()
    data[field] = "too-short"
    write_manifest(tmp_path / "manifest.json", data)

    with pytest.raises(ArtifactMismatchError, match=f"invalid {field}"):
        EngineManifest.load(tmp_path / "manifest.json")


@pytest.mark.parametrize(
    "field",
    ["checkpoint_sha256", "onnx_sha256", "engine_sha256"],
)
def test_manifest_requires_lowercase_hex_hashes(tmp_path, field):
    data = manifest_data()
    data[field] = "z" * 64
    write_manifest(tmp_path / "manifest.json", data)

    with pytest.raises(ArtifactMismatchError, match=f"invalid {field}"):
        EngineManifest.load(tmp_path / "manifest.json")


@pytest.mark.parametrize(
    "field",
    [
        "gpu",
        "compute_capability",
        "python",
        "pytorch",
        "cuda",
        "tensorrt",
    ],
)
def test_manifest_requires_environment_metadata(tmp_path, field):
    data = manifest_data()
    data["environment"][field] = ""
    write_manifest(tmp_path / "manifest.json", data)

    with pytest.raises(ArtifactMismatchError, match="engine environment"):
        EngineManifest.load(tmp_path / "manifest.json")


def test_manifest_requires_named_profiles(tmp_path):
    data = manifest_data()
    data["profiles"][0]["name"] = ""
    write_manifest(tmp_path / "manifest.json", data)

    with pytest.raises(ArtifactMismatchError, match="engine profile"):
        EngineManifest.load(tmp_path / "manifest.json")


def test_manifest_rejects_non_positive_profile_dimension(tmp_path):
    data = manifest_data()
    data["profiles"][0]["min"][0] = 0
    write_manifest(tmp_path / "manifest.json", data)

    with pytest.raises(ArtifactMismatchError, match="engine profile shape"):
        EngineManifest.load(tmp_path / "manifest.json")


def test_tensorrt_backend_validates_engine_before_deserialization(
    tmp_path,
    monkeypatch,
):
    engine = tmp_path / "encoder.plan"
    engine.write_bytes(b"engine-a")
    write_manifest(
        tmp_path / "manifest.json",
        manifest_data(
            engine_sha256=hashlib.sha256(b"engine-b").hexdigest(),
        ),
    )
    deserialize_calls = []

    class FakeLogger:
        WARNING = 1

        def __init__(self, severity):
            self.severity = severity

    class FakeRuntime:
        def __init__(self, logger):
            self.logger = logger

        def deserialize_cuda_engine(self, serialized):
            deserialize_calls.append(serialized)
            raise AssertionError("corrupt engine reached TensorRT")

    fake_trt = SimpleNamespace(Logger=FakeLogger, Runtime=FakeRuntime)
    monkeypatch.setattr(tensorrt_backend.sys, "platform", "linux")
    monkeypatch.setattr(
        tensorrt_backend.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        tensorrt_backend.importlib.util,
        "find_spec",
        lambda name: object(),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

    with pytest.raises(ArtifactMismatchError, match="engine SHA-256"):
        TensorRTEncoderBackend(tmp_path)

    assert deserialize_calls == []


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


def test_build_writes_schema_v2_engine_metadata(tmp_path, monkeypatch):
    module = load_build_module()
    onnx_path, checkpoint, profiles = write_valid_preflight_bundle(tmp_path)
    output_dir = tmp_path / "engine"
    output_dir.mkdir()
    engine_path = output_dir / "encoder.plan"
    engine_path.write_bytes(b"serialized-engine")
    properties = SimpleNamespace(
        name="Test GPU",
        major=8,
        minor=9,
    )
    monkeypatch.setattr(
        module.torch.cuda,
        "current_device",
        lambda: 0,
    )
    monkeypatch.setattr(
        module.torch.cuda,
        "get_device_properties",
        lambda device: properties,
    )

    module.write_manifest(
        output_dir,
        module.load_profile(profiles),
        checkpoint,
        onnx_path,
        engine_path=engine_path,
        trt_version="10.14.1",
    )

    data = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert data["schema_version"] == 2
    assert data["engine_sha256"] == hashlib.sha256(
        b"serialized-engine"
    ).hexdigest()
    assert data["tensor_dtypes"] == manifest_data()["tensor_dtypes"]
    assert data["environment"]["gpu"] == "Test GPU"
    assert data["environment"]["compute_capability"] == "8.9"
    assert all(
        isinstance(value, str) and value
        for value in data["environment"].values()
    )


def test_build_removes_stale_manifest_when_serialization_fails(
    tmp_path,
    monkeypatch,
):
    module = load_build_module()
    _, checkpoint, profiles = write_valid_preflight_bundle(tmp_path)
    output_dir = tmp_path / "engine"
    output_dir.mkdir()
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text("stale", encoding="utf-8")

    class FakeLogger:
        INFO = 1

        def __init__(self, severity):
            self.severity = severity

    class FakeTensor:
        def __init__(self, name):
            self.name = name
            self.dtype = None

    class FakeNetwork:
        def __init__(self):
            self.inputs = [
                FakeTensor("features"),
                FakeTensor("feature_lengths"),
            ]
            self.outputs = [
                FakeTensor("encoder_outputs"),
                FakeTensor("encoder_lengths"),
                FakeTensor("encoder_mask"),
            ]
            self.num_inputs = len(self.inputs)
            self.num_outputs = len(self.outputs)

        def get_input(self, index):
            return self.inputs[index]

        def get_output(self, index):
            return self.outputs[index]

    class FakeParser:
        def __init__(self, network, logger):
            self.network = network
            self.logger = logger

        def parse_from_file(self, path):
            return True

    class FakeProfile:
        def set_shape(self, name, minimum, optimum, maximum):
            return True

    class FakeConfig:
        def set_flag(self, flag):
            self.flag = flag

        def add_optimization_profile(self, profile):
            self.profile = profile

    class FakeBuilder:
        def __init__(self, logger):
            self.logger = logger

        def create_network(self, flags):
            return FakeNetwork()

        def create_builder_config(self):
            return FakeConfig()

        def create_optimization_profile(self):
            return FakeProfile()

        def build_serialized_network(self, network, config):
            return None

    fake_trt = SimpleNamespace(
        Logger=FakeLogger,
        Builder=FakeBuilder,
        NetworkDefinitionCreationFlag=SimpleNamespace(
            EXPLICIT_BATCH=0,
        ),
        OnnxParser=FakeParser,
        BuilderFlag=SimpleNamespace(FP16=0),
        float16="float16",
        int64="int64",
        uint8="uint8",
    )
    monkeypatch.setattr(
        module,
        "preflight_artifacts",
        lambda *args: {},
    )
    monkeypatch.setattr(module, "require_tensorrt", lambda: fake_trt)

    with pytest.raises(RuntimeError, match="empty serialized engine"):
        module.build_engine(
            tmp_path / "encoder.onnx",
            output_dir,
            profiles,
            checkpoint,
        )

    assert not manifest_path.exists()


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
    artifacts = report["provenance"]["input_artifacts"]
    assert set(artifacts) == {"checkpoint", "onnx_bundle", "profiles"}
    assert artifacts["checkpoint"]["sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert artifacts["profiles"]["sha256"] == hashlib.sha256(
        profiles.read_bytes()
    ).hexdigest()
    assert artifacts["onnx_bundle"]["sha256"] == report[
        "onnx_bundle_sha256"
    ]
    assert report["provenance"]["arguments"] == {
        "onnx": str(onnx_path),
        "checkpoint": str(checkpoint),
        "profiles": str(profiles),
    }
    json.dumps(report["provenance"])


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
