import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper


SCRIPT = Path("runtime/fireredlid/export_encoder_onnx.py")
VERIFY_SCRIPT = Path("runtime/fireredlid/verify.py")


class TinyEncoder(torch.nn.Module):
    def forward(self, features, lengths):
        mask = (
            torch.arange(features.size(1))[None, :] < lengths[:, None]
        ).unsqueeze(1).to(torch.uint8)
        return features * 2.0, lengths, mask


def load_export_module():
    spec = importlib.util.spec_from_file_location(
        "fireredlid_export",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_verify_module():
    spec = importlib.util.spec_from_file_location(
        "fireredlid_verify",
        VERIFY_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_supports_dynamic_batch_and_time(tmp_path):
    module = load_export_module()
    path = tmp_path / "encoder.onnx"
    module.export_encoder(
        TinyEncoder().eval(),
        path,
        torch.zeros(1, 4, 80),
        torch.tensor([4]),
    )
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )

    outputs = session.run(
        None,
        {
            "features": torch.zeros(2, 7, 80).numpy(),
            "feature_lengths": torch.tensor([5, 7]).numpy(),
        },
    )

    assert outputs[0].shape == (2, 7, 80)
    assert outputs[1].shape == (2,)
    assert outputs[2].shape == (2, 1, 7)
    np.testing.assert_array_equal(outputs[1], np.array([5, 7]))
    np.testing.assert_array_equal(
        outputs[2],
        np.array(
            [
                [[1, 1, 1, 1, 1, 0, 0]],
                [[1, 1, 1, 1, 1, 1, 1]],
            ],
            dtype=np.uint8,
        ),
    )


def test_external_data_stays_in_data_directory(tmp_path):
    module = load_export_module()
    path = tmp_path / "encoder.onnx"
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    staged_path = data_directory / path.name
    weight = numpy_helper.from_array(
        np.eye(2, dtype=np.float32),
        name="weight",
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["features", "weight"], ["outputs"])],
        "external-data-test",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info("outputs", TensorProto.FLOAT, [None, 2])],
        [weight],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.save_model(
        model,
        staged_path,
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        size_threshold=0,
    )
    external_model = onnx.load(staged_path, load_external_data=False)
    location = next(
        item.value
        for item in external_model.graph.initializer[0].external_data
        if item.key == "location"
    )
    external_path = data_directory / location
    assert external_path.is_file()
    external_inode = external_path.stat().st_ino

    module.finalize_external_data_layout(staged_path, path)

    relocated_model = onnx.load(path, load_external_data=False)
    relocated_location = next(
        item.value
        for item in relocated_model.graph.initializer[0].external_data
        if item.key == "location"
    )
    assert relocated_location == f"data/{location}"
    assert external_path.is_file()
    assert external_path.stat().st_ino == external_inode
    assert not staged_path.exists()
    onnx.checker.check_model(path)


def test_verify_onnx_outputs_checks_all_three_outputs(tmp_path):
    export_module = load_export_module()
    verify_module = load_verify_module()
    encoder = TinyEncoder().eval()
    path = tmp_path / "encoder.onnx"
    export_module.export_encoder(
        encoder,
        path,
        torch.zeros(1, 4, 80),
        torch.tensor([4]),
    )
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )

    report = verify_module.verify_onnx_outputs(
        encoder,
        session,
        torch.ones(2, 7, 80),
        torch.tensor([5, 7]),
    )

    assert report["max_abs_error"] == 0.0


def test_verify_backend_outputs_checks_all_three_outputs():
    verify_module = load_verify_module()
    encoder = TinyEncoder().eval()
    features = torch.ones(2, 7, 80)
    lengths = torch.tensor([5, 7])

    class Backend:
        def encode(self, input_features, input_lengths):
            outputs, output_lengths, mask = encoder(
                input_features,
                input_lengths,
            )
            return SimpleNamespace(
                outputs=outputs,
                lengths=output_lengths,
                mask=mask,
            )

    report = verify_module.verify_backend_outputs(
        encoder,
        Backend(),
        features,
        lengths,
    )

    assert report["max_abs_error"] == 0.0
