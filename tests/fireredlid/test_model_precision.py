import pytest
import torch

from fireredasr2s.fireredlid.models.fireredlid_aed import FireRedLidAed
from fireredasr2s.fireredlid.models.module.transformer_decoder import (
    TransformerDecoder,
)


class HalfEncoder(torch.nn.Module):
    def forward(self, padded_input, input_lengths):
        batch = padded_input.size(0)
        outputs = torch.ones(batch, 2, 4, dtype=torch.float16)
        mask = torch.ones(batch, 1, 2, dtype=torch.uint8)
        return outputs, input_lengths, mask


class RecordingDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.encoder_dtype = None

    def batch_beam_search(self, encoder_outputs, *args):
        self.encoder_dtype = encoder_outputs.dtype
        return [[{"yseq": torch.tensor([1]), "confidence": torch.tensor(1.0)}]]


def test_decoder_receives_encoder_outputs_in_its_own_precision():
    model = FireRedLidAed.__new__(FireRedLidAed)
    torch.nn.Module.__init__(model)
    model.encoder = HalfEncoder()
    model.lid_decoder = RecordingDecoder()

    model.process(
        torch.ones(1, 4, 80, dtype=torch.float16),
        torch.tensor([4]),
    )

    assert model.lid_decoder.encoder_dtype == torch.float32


def test_decoder_reports_non_finite_confidence_explicitly():
    with pytest.raises(
        FloatingPointError,
        match="non-finite",
    ) as error:
        TransformerDecoder.validate_token_confidences(
            torch.tensor(
                [[0.9], [float("nan")], [0.8], [0.7]],
                dtype=torch.float32,
            ),
            beam_size=2,
        )

    assert error.value.sample_indices == (0,)


def test_decoder_clamps_finite_confidence_roundoff():
    confidences = TransformerDecoder.validate_token_confidences(
        torch.tensor([-1e-7, 1.0 + 1e-7], dtype=torch.float32)
    )

    assert torch.equal(confidences, torch.tensor([0.0, 1.0]))


def test_decoder_rejects_eos_only_empty_confidence():
    with pytest.raises(
        FloatingPointError,
        match="no language token",
    ) as error:
        TransformerDecoder.average_token_confidence(
            torch.tensor([], dtype=torch.float32),
            sample_index=3,
        )

    assert error.value.sample_indices == (3,)


def test_decoder_reports_all_eos_only_samples_in_a_batch():
    class EosProjection(torch.nn.Module):
        def forward(self, inputs):
            logits = torch.zeros(
                inputs.size(0),
                3,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            logits[:, 1] = 100.0
            return logits

    decoder = TransformerDecoder(
        sos_id=0,
        eos_id=1,
        pad_id=2,
        odim=3,
        n_layers=0,
        n_head=1,
        d_model=4,
        residual_dropout=0.0,
    )
    decoder.tgt_word_prj = EosProjection()

    with pytest.raises(
        FloatingPointError,
        match="no language token",
    ) as error:
        decoder.batch_beam_search(
            torch.zeros(4, 2, 4),
            torch.ones(4, 1, 2, dtype=torch.uint8),
            beam_size=1,
            nbest=1,
            decode_max_len=1,
        )

    assert error.value.sample_indices == (0, 1, 2, 3)
