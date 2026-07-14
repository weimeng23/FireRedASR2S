import torch

from fireredasr2s.fireredlid.models.module.conformer_encoder import ConformerEncoder


class MaskModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConformerEncoder(
            idim=80,
            n_layers=0,
            n_head=2,
            d_model=8,
            residual_dropout=0.0,
            dropout_rate=0.0,
            kernel_size=3,
            pe_maxlen=64,
        )

    def forward(self, features, lengths):
        return self.encoder.padding_position_is_0(features, lengths)


def test_padding_mask_is_exact():
    module = MaskModule()
    features = torch.zeros(2, 5, 80)
    lengths = torch.tensor([3, 5])

    actual = module(features, lengths)

    expected = torch.tensor(
        [
            [[1, 1, 1, 0, 0]],
            [[1, 1, 1, 1, 1]],
        ],
        dtype=torch.uint8,
    )
    assert torch.equal(actual, expected)


def test_padding_mask_is_torch_exportable():
    module = MaskModule().eval()
    features = torch.zeros(2, 5, 80)
    lengths = torch.tensor([3, 5])

    exported = torch.export.export(module, (features, lengths))

    assert torch.equal(
        exported.module()(features, lengths),
        module(features, lengths),
    )
