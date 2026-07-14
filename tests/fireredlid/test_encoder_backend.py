import torch

from fireredasr2s.fireredlid.runtime.encoder_backend import (
    CompatibleEncoderAdapter,
)
from fireredasr2s.fireredlid.runtime.pytorch_backend import (
    CompileEncoderBackend,
    EagerEncoderBackend,
)


class FakeEncoder(torch.nn.Module):
    def forward(self, features, lengths):
        mask = (
            torch.arange(features.size(1))[None, :] < lengths[:, None]
        ).unsqueeze(1).to(torch.uint8)
        return features + 1.0, lengths + 10, mask


def test_adapter_preserves_official_three_output_contract():
    adapter = CompatibleEncoderAdapter(EagerEncoderBackend(FakeEncoder()))
    features = torch.zeros(2, 4, 3)

    outputs, lengths, mask = adapter(features, torch.tensor([2, 4]))

    assert torch.equal(outputs, torch.ones_like(features))
    assert lengths.tolist() == [12, 14]
    assert mask.dtype == torch.uint8


def test_compile_backend_uses_dynamic_fullgraph(monkeypatch):
    captured = {}

    def fake_compile(module, **kwargs):
        captured.update(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)

    CompileEncoderBackend(FakeEncoder(), profile="latency")

    assert captured == {
        "mode": "reduce-overhead",
        "dynamic": True,
        "fullgraph": True,
    }
