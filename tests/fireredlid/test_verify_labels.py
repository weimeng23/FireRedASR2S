import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fireredasr2s.fireredlid import lid as lid_module


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


def test_run_backend_keeps_verification_batching_in_the_caller(monkeypatch):
    module = load_verify_labels_module()

    class FakeModel:
        active_backend = "tensorrt"

        def __init__(self):
            self.calls = []

        def process(self, uttids, wavs):
            self.calls.append((list(uttids), list(wavs)))
            return [
                {
                    "uttid": uttids[0],
                    "lang": "en",
                    "confidence": 0.9,
                }
            ]

    model = FakeModel()
    monkeypatch.setattr(
        lid_module.FireRedLid,
        "from_pretrained",
        staticmethod(lambda unused_dir, unused_config: model),
    )

    results = module.run_backend(
        [
            {"uttid": "one", "wav": "one.wav"},
            {"uttid": "two", "wav": "two.wav"},
        ],
        model_dir="/model",
        backend="tensorrt",
        engine_dir="/engine",
    )

    assert model.calls == [
        (["one"], ["one.wav"]),
        (["two"], ["two.wav"]),
    ]
    assert [result["uttid"] for result in results] == ["one", "two"]


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


def test_main_report_includes_common_provenance_and_real_input_hashes(
    tmp_path,
    monkeypatch,
):
    module = load_verify_labels_module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_files = {
        "checkpoint": model_dir / "model.pth.tar",
        "cmvn": model_dir / "cmvn.ark",
        "dictionary": model_dir / "dict.txt",
    }
    for name, path in model_files.items():
        path.write_bytes(name.encode("utf-8"))
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        '{"uttid":"a","wav":"a.wav"}\n',
        encoding="utf-8",
    )
    report_path = tmp_path / "verify.labels.json"
    args = SimpleNamespace(
        model_dir=str(model_dir),
        manifest=str(manifest),
        candidate_backend="compile",
        engine_dir=None,
        confidence_atol=0.005,
        report=str(report_path),
    )
    result = [{"uttid": "a", "lang": "en", "confidence": 0.9}]
    monkeypatch.setattr(module, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        module,
        "run_backend",
        lambda *unused_args, **unused_kwargs: result,
    )
    monkeypatch.setattr(module, "_environment", lambda: {"gpu": "test"})

    assert module.main([]) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifacts = report["provenance"]["input_artifacts"]
    assert set(artifacts) == {
        "checkpoint",
        "cmvn",
        "dictionary",
        "audio_manifest",
    }
    for name, path in model_files.items():
        assert artifacts[name]["sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    assert artifacts["audio_manifest"]["sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert report["provenance"]["arguments"]["candidate_backend"] == "compile"
    json.dumps(report["provenance"])
