import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path("runtime/fireredlid/verify_labels.py")


def load_verify_labels_module():
    spec = importlib.util.spec_from_file_location(
        "fireredlid_verify_labels",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_results_accepts_exact_labels_with_small_confidence_delta():
    module = load_verify_labels_module()

    report = module.compare_results(
        [{"uttid": "a", "lang": "en", "confidence": 0.900}],
        [{"uttid": "a", "lang": "en", "confidence": 0.904}],
        confidence_atol=0.005,
    )

    assert report["passed"] is True
    assert report["label_mismatches"] == []
    assert report["confidence_mismatches"] == []


def test_compare_results_rejects_any_label_mismatch():
    module = load_verify_labels_module()

    report = module.compare_results(
        [{"uttid": "a", "lang": "en", "confidence": 0.900}],
        [
            {
                "uttid": "a",
                "lang": "zh mandarin",
                "confidence": 0.900,
            }
        ],
        confidence_atol=0.005,
    )

    assert report["passed"] is False
    assert report["label_mismatches"] == [
        {
            "uttid": "a",
            "baseline_lang": "en",
            "candidate_lang": "zh mandarin",
        }
    ]


def test_compare_results_accepts_confidence_delta_exactly_at_threshold():
    module = load_verify_labels_module()

    report = module.compare_results(
        [{"uttid": "a", "lang": "en", "confidence": 0.900}],
        [{"uttid": "a", "lang": "en", "confidence": 0.905}],
        confidence_atol=0.005,
    )

    assert report["passed"] is True
    assert report["max_confidence_abs_error"] == 0.005


def test_compare_results_rejects_confidence_delta_above_threshold():
    module = load_verify_labels_module()

    report = module.compare_results(
        [{"uttid": "a", "lang": "en", "confidence": 0.900}],
        [{"uttid": "a", "lang": "en", "confidence": 0.906}],
        confidence_atol=0.005,
    )

    assert report["passed"] is False
    assert report["max_confidence_abs_error"] == 0.006
    assert report["confidence_mismatches"] == [
        {
            "uttid": "a",
            "baseline_confidence": 0.900,
            "candidate_confidence": 0.906,
            "abs_error": 0.006,
        }
    ]


def test_compare_results_reports_missing_uttids_from_each_side():
    module = load_verify_labels_module()

    report = module.compare_results(
        [
            {"uttid": "a", "lang": "en", "confidence": 0.900},
            {"uttid": "shared", "lang": "en", "confidence": 0.800},
        ],
        [
            {"uttid": "shared", "lang": "en", "confidence": 0.800},
            {"uttid": "z", "lang": "en", "confidence": 0.700},
        ],
        confidence_atol=0.005,
    )

    assert report["passed"] is False
    assert report["missing_in_baseline"] == ["z"]
    assert report["missing_in_candidate"] == ["a"]
    assert report["counts"] == {
        "baseline": 2,
        "candidate": 2,
        "compared": 1,
    }


def test_compare_results_rejects_duplicate_uttids_from_either_side():
    module = load_verify_labels_module()

    report = module.compare_results(
        [
            {"uttid": "a", "lang": "en", "confidence": 0.900},
            {"uttid": "a", "lang": "en", "confidence": 0.900},
        ],
        [
            {"uttid": "a", "lang": "en", "confidence": 0.900},
            {"uttid": "b", "lang": "en", "confidence": 0.800},
            {"uttid": "b", "lang": "en", "confidence": 0.800},
        ],
        confidence_atol=0.005,
    )

    assert report["passed"] is False
    assert report["duplicate_uttids"] == {
        "baseline": ["a"],
        "candidate": ["b"],
    }
    assert report["missing_in_baseline"] == ["b"]
    assert report["missing_in_candidate"] == []
    assert report["counts"]["compared"] == 0


def test_compare_results_is_stable_across_input_order():
    module = load_verify_labels_module()
    baseline = [
        {"uttid": "b", "lang": "en", "confidence": 0.900},
        {"uttid": "a", "lang": "en", "confidence": 0.900},
    ]
    candidate = [
        {"uttid": "b", "lang": "zh mandarin", "confidence": 0.900},
        {"uttid": "a", "lang": "en", "confidence": 0.910},
    ]

    report = module.compare_results(
        baseline,
        candidate,
        confidence_atol=0.005,
    )
    reordered = module.compare_results(
        list(reversed(baseline)),
        list(reversed(candidate)),
        confidence_atol=0.005,
    )

    assert report == reordered
    assert report["passed"] is False
    assert report["label_mismatches"][0]["uttid"] == "b"
    assert report["confidence_mismatches"][0]["uttid"] == "a"


@pytest.mark.parametrize("confidence_atol", ["nan", "inf"])
def test_parse_args_rejects_non_finite_confidence_atol(confidence_atol):
    module = load_verify_labels_module()

    with pytest.raises(SystemExit) as error:
        module.parse_args(
            [
                "--model-dir",
                "FireRedLID",
                "--manifest",
                "manifest.jsonl",
                "--candidate-backend",
                "compile",
                "--confidence-atol",
                confidence_atol,
                "--report",
                "verify.labels.json",
            ]
        )

    assert error.value.code == 2
