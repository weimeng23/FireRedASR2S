import torch

from .encoder_backend import EncoderBackend, EncoderResult


class EagerEncoderBackend(EncoderBackend):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def encode(self, features, feature_lengths):
        outputs, lengths, mask = self.encoder(features, feature_lengths)
        return EncoderResult(
            outputs=outputs,
            lengths=lengths,
            mask=mask,
        )


class CompileEncoderBackend(EagerEncoderBackend):
    def __init__(self, encoder: torch.nn.Module, profile: str):
        mode = "reduce-overhead" if profile == "latency" else "max-autotune"
        compiled = torch.compile(
            encoder,
            mode=mode,
            dynamic=True,
            fullgraph=True,
        )
        super().__init__(compiled)
