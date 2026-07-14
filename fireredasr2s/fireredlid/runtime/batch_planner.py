from bisect import bisect_left
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class FeatureItem:
    index: int
    uttid: str
    wav_input: object
    feature: Tensor
    duration_s: float
    processed_duration_s: float
    truncated: bool


@dataclass(frozen=True)
class PlannedBatch:
    items: tuple[FeatureItem, ...]
    padded_features: Tensor
    feature_lengths: Tensor


def pad_features(
    features: Sequence[Tensor], pad_value: float = 0.0
) -> Tensor:
    if not features:
        raise ValueError("features must not be empty")
    max_frames = max(feature.size(0) for feature in features)
    feature_dim = features[0].size(1)
    padded = features[0].new_full(
        (len(features), max_frames, feature_dim), pad_value
    )
    for index, feature in enumerate(features):
        padded[index, : feature.size(0)] = feature
    return padded


class BatchPlanner:
    VALID_STRATEGIES = {"none", "bucket", "auto"}

    def __init__(
        self,
        strategy: str,
        max_sub_batch_size: int | None,
        bucket_boundaries_s: tuple[float, ...] = (5.0, 15.0, 30.0, 60.0),
        auto_min_saving_ratio: float = 0.20,
    ):
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"unsupported batch strategy: {strategy}")
        if max_sub_batch_size is not None and max_sub_batch_size < 1:
            raise ValueError("max_sub_batch_size must be positive")
        self.strategy = strategy
        self.max_sub_batch_size = max_sub_batch_size
        self.bucket_boundaries_s = bucket_boundaries_s
        self.auto_min_saving_ratio = auto_min_saving_ratio

    def _chunk(
        self, items: Sequence[FeatureItem]
    ) -> list[list[FeatureItem]]:
        size = self.max_sub_batch_size or max(1, len(items))
        return [
            list(items[start : start + size])
            for start in range(0, len(items), size)
        ]

    def _bucket_groups(
        self, items: Sequence[FeatureItem]
    ) -> list[list[FeatureItem]]:
        buckets: dict[int, list[FeatureItem]] = {}
        for item in items:
            key = bisect_left(
                self.bucket_boundaries_s, item.processed_duration_s
            )
            buckets.setdefault(key, []).append(item)
        groups = []
        for key in sorted(buckets):
            groups.extend(self._chunk(buckets[key]))
        return groups

    @staticmethod
    def _padded_frames(groups: Sequence[Sequence[FeatureItem]]) -> int:
        return sum(
            len(group) * max(item.feature.size(0) for item in group)
            for group in groups
        )

    def plan(self, items: Sequence[FeatureItem]) -> list[PlannedBatch]:
        if not items:
            return []
        direct = self._chunk(items)
        bucketed = self._bucket_groups(items)
        groups = direct
        if self.strategy == "bucket":
            groups = bucketed
        elif self.strategy == "auto":
            saving = 1.0 - (
                self._padded_frames(bucketed) / self._padded_frames(direct)
            )
            if saving >= self.auto_min_saving_ratio:
                groups = bucketed
        return [
            PlannedBatch(
                items=tuple(group),
                padded_features=pad_features(
                    [item.feature for item in group]
                ),
                feature_lengths=torch.tensor(
                    [item.feature.size(0) for item in group],
                    dtype=torch.long,
                ),
            )
            for group in groups
        ]
