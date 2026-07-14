from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EncoderResult:
    outputs: Tensor
    lengths: Tensor
    mask: Tensor


class EncoderBackend(torch.nn.Module, ABC):
    @abstractmethod
    def encode(
        self, features: Tensor, feature_lengths: Tensor
    ) -> EncoderResult:
        raise NotImplementedError


class CompatibleEncoderAdapter(torch.nn.Module):
    def __init__(self, backend: EncoderBackend):
        super().__init__()
        self.backend = backend

    def forward(self, features: Tensor, feature_lengths: Tensor):
        result = self.backend.encode(features, feature_lengths)
        return result.outputs, result.lengths, result.mask
