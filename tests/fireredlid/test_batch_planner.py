import torch

from fireredasr2s.fireredlid.runtime.batch_planner import (
    BatchPlanner,
    FeatureItem,
)


def make_item(index: int, frames: int, duration_s: float) -> FeatureItem:
    return FeatureItem(
        index=index,
        uttid=f"utt-{index}",
        wav_input=f"{index}.wav",
        feature=torch.full((frames, 80), float(index + 1)),
        duration_s=duration_s,
        processed_duration_s=duration_s,
        truncated=False,
    )


def batch_indices(plans):
    return [[item.index for item in plan.items] for plan in plans]


def test_none_preserves_order_and_chunks():
    items = [
        make_item(0, 4, 1),
        make_item(1, 2, 1),
        make_item(2, 3, 1),
    ]

    plans = BatchPlanner("none", max_sub_batch_size=2).plan(items)

    assert batch_indices(plans) == [[0, 1], [2]]
    assert plans[0].padded_features.shape == (2, 4, 80)
    assert plans[0].feature_lengths.tolist() == [4, 2]
    assert torch.count_nonzero(plans[0].padded_features[1, 2:]) == 0


def test_bucket_groups_by_processed_duration():
    items = [
        make_item(0, 3000, 30),
        make_item(1, 400, 4),
        make_item(2, 1000, 10),
    ]

    plans = BatchPlanner("bucket", max_sub_batch_size=8).plan(items)

    assert batch_indices(plans) == [[1], [2], [0]]


def test_auto_buckets_only_when_saving_reaches_twenty_percent():
    mixed = [
        make_item(0, 6000, 60),
        make_item(1, 500, 5),
        make_item(2, 500, 5),
    ]
    close = [
        make_item(0, 5000, 50),
        make_item(1, 5500, 55),
    ]
    planner = BatchPlanner("auto", max_sub_batch_size=8)

    assert batch_indices(planner.plan(mixed)) == [[1, 2], [0]]
    assert batch_indices(planner.plan(close)) == [[0, 1]]
