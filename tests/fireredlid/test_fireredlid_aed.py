from contextlib import contextmanager

import torch

from fireredasr2s.fireredlid.models.fireredlid_aed import FireRedLidAed


class RecordingStageRecorder:
    def __init__(self):
        self.names = []

    @contextmanager
    def measure(self, name):
        self.names.append(name)
        yield


class FakeEncoder(torch.nn.Module):
    def forward(self, features, lengths):
        mask = torch.ones(
            features.size(0),
            1,
            features.size(1),
            dtype=torch.uint8,
        )
        return features, lengths, mask


class FakeDecoder(torch.nn.Module):
    def batch_beam_search(self, *args):
        return [[{"yseq": torch.tensor([1])}]]


def test_process_records_encoder_and_decoder_stages():
    model = FireRedLidAed.__new__(FireRedLidAed)
    torch.nn.Module.__init__(model)
    model.encoder = FakeEncoder()
    model.lid_decoder = FakeDecoder()
    recorder = RecordingStageRecorder()

    model.process(
        torch.zeros(1, 4, 3),
        torch.tensor([4]),
        stage_recorder=recorder,
    )

    assert recorder.names == ["encoder", "decoder"]
