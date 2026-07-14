import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch


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
