import importlib.util
import json
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path("runtime/fireredlid/run_benchmark_matrix.py")


def load_benchmark_matrix_module():
    spec = importlib.util.spec_from_file_location(
        "fireredlid_benchmark_matrix",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_args(**overrides):
    values = {
        "model_dir": Path("FireRedLID"),
        "manifest": Path("manifest.jsonl"),
        "engine_dir": Path("engine"),
        "profile": "latency",
        "backends": ["eager", "compile", "tensorrt"],
        "scopes": ["encoder", "model", "end-to-end"],
        "device": "cuda",
        "precision": "fp16",
        "output_dir": Path("matrix"),
        "execute": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def option_value(command, option):
    return command[command.index(option) + 1]


def replace_subprocess_run(module, monkeypatch, fake_run):
    subprocess_proxy = SimpleNamespace(
        run=fake_run,
        check_output=subprocess.check_output,
        CalledProcessError=subprocess.CalledProcessError,
    )
    monkeypatch.setattr(module, "subprocess", subprocess_proxy)


def execution_argv(tmp_path, *extra):
    model_dir = tmp_path / "model"
    manifest = tmp_path / "manifest.jsonl"
    engine_dir = tmp_path / "engine"
    model_dir.mkdir()
    manifest.write_text('{"uttid":"a","wav":"a.wav"}\n', encoding="utf-8")
    engine_dir.mkdir()
    return [
        "--model-dir",
        str(model_dir),
        "--manifest",
        str(manifest),
        "--engine-dir",
        str(engine_dir),
        "--profile",
        "latency",
        "--output-dir",
        str(tmp_path / "matrix"),
        *extra,
    ]


def test_latency_matrix_has_nine_commands_with_profile_defaults(tmp_path):
    module = load_benchmark_matrix_module()

    commands = module.build_commands(
        make_args(output_dir=tmp_path, profile="latency")
    )

    assert len(commands) == 9
    assert [
        (option_value(command, "--backend"), option_value(command, "--scope"))
        for command in commands
    ] == [
        (backend, scope)
        for backend in ["eager", "compile", "tensorrt"]
        for scope in ["encoder", "model", "end-to-end"]
    ]
    assert all(
        option_value(command, "--logical-batch-size") == "1"
        for command in commands
    )
    assert all(
        option_value(command, "--batch-strategy") == "none"
        for command in commands
    )
    assert all(option_value(command, "--warmup") == "5" for command in commands)
    assert all(option_value(command, "--iterations") == "50" for command in commands)


def test_throughput_matrix_uses_profile_defaults(tmp_path):
    module = load_benchmark_matrix_module()

    commands = module.build_commands(
        make_args(
            output_dir=tmp_path,
            profile="throughput",
            backends=["eager"],
            scopes=["encoder"],
        )
    )

    assert len(commands) == 1
    command = commands[0]
    assert option_value(command, "--logical-batch-size") == "100"
    assert option_value(command, "--batch-strategy") == "auto"
    assert option_value(command, "--warmup") == "3"
    assert option_value(command, "--iterations") == "20"


def test_command_order_and_output_names_are_deterministic(tmp_path):
    module = load_benchmark_matrix_module()
    args = make_args(output_dir=tmp_path)

    first = module.build_commands(args)
    second = module.build_commands(args)

    assert first == second
    assert [option_value(command, "--output") for command in first] == [
        str(tmp_path / f"benchmark.{backend}.latency.{scope}.json")
        for backend in ["eager", "compile", "tensorrt"]
        for scope in ["encoder", "model", "end-to-end"]
    ]


def test_tensorrt_commands_always_include_engine_dir(tmp_path):
    module = load_benchmark_matrix_module()
    engine_dir = tmp_path / "engine"

    commands = module.build_commands(
        make_args(output_dir=tmp_path, engine_dir=engine_dir)
    )

    trt_commands = [
        command
        for command in commands
        if option_value(command, "--backend") == "tensorrt"
    ]
    assert trt_commands
    assert all("--engine-dir" in command for command in trt_commands)
    assert all(
        option_value(command, "--engine-dir") == str(engine_dir)
        for command in trt_commands
    )


def test_default_tensorrt_selection_requires_engine_dir_argument():
    module = load_benchmark_matrix_module()

    with pytest.raises(SystemExit) as error:
        module.parse_args(
            [
                "--model-dir",
                "missing-model",
                "--manifest",
                "missing-manifest.jsonl",
                "--profile",
                "latency",
                "--output-dir",
                "matrix",
            ]
        )

    assert error.value.code == 2


def test_dry_run_allows_missing_paths_and_never_launches_subprocess(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = load_benchmark_matrix_module()

    def unexpected_run(*args, **kwargs):
        raise AssertionError("dry-run must not launch subprocesses")

    replace_subprocess_run(module, monkeypatch, unexpected_run)
    output_dir = tmp_path / "matrix output"
    result = module.main(
        [
            "--model-dir",
            str(tmp_path / "missing model"),
            "--manifest",
            str(tmp_path / "missing manifest.jsonl"),
            "--engine-dir",
            str(tmp_path / "missing engine"),
            "--profile",
            "latency",
            "--output-dir",
            str(output_dir),
        ]
    )

    printed = capsys.readouterr().out.splitlines()
    first_command = shlex.split(printed[0])
    assert result == 0
    assert len(printed) == 9
    assert option_value(first_command, "--model-dir") == str(
        tmp_path / "missing model"
    )
    assert option_value(first_command, "--manifest") == str(
        tmp_path / "missing manifest.jsonl"
    )
    assert option_value(first_command, "--output") == str(
        output_dir / "benchmark.eager.latency.encoder.json"
    )
    assert not output_dir.exists()


@pytest.mark.parametrize("missing", ["model", "manifest", "engine"])
def test_execute_validates_all_required_paths_before_launch(
    tmp_path,
    monkeypatch,
    missing,
):
    module = load_benchmark_matrix_module()
    argv = execution_argv(tmp_path, "--execute")
    paths = {
        "model": Path(argv[argv.index("--model-dir") + 1]),
        "manifest": Path(argv[argv.index("--manifest") + 1]),
        "engine": Path(argv[argv.index("--engine-dir") + 1]),
    }
    path = paths[missing]
    if path.is_dir():
        path.rmdir()
    else:
        path.unlink()
    calls = []

    def fake_run(command, check):
        calls.append((command, check))
        return subprocess.CompletedProcess(command, 0)

    replace_subprocess_run(module, monkeypatch, fake_run)

    with pytest.raises(FileNotFoundError, match=missing):
        module.main(argv)

    assert calls == []


def test_execute_uses_list_commands_and_writes_complete_index(
    tmp_path,
    monkeypatch,
):
    module = load_benchmark_matrix_module()
    calls = []

    def fake_run(command, check):
        calls.append((command, check))
        output = Path(option_value(command, "--output"))
        output.write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    replace_subprocess_run(module, monkeypatch, fake_run)
    argv = execution_argv(
        tmp_path,
        "--backends",
        "eager",
        "--scopes",
        "encoder",
        "model",
        "--execute",
    )

    result = module.main(argv)

    assert result == 0
    assert len(calls) == 2
    assert all(
        isinstance(command, list) and check is False
        for command, check in calls
    )
    output_dir = tmp_path / "matrix"
    index = json.loads((output_dir / "matrix.index.json").read_text())
    assert index["schema_version"] == 1
    assert index["profile"] == "latency"
    assert set(index["environment"]) >= {
        "platform",
        "python",
        "pytorch",
        "cuda",
        "gpu",
    }
    assert len(index["commit_hash"]) == 40
    assert index["runs"] == [
        {
            "command": command,
            "output_path": option_value(command, "--output"),
            "return_code": 0,
        }
        for command, _ in calls
    ]
    assert all(Path(run["output_path"]).is_file() for run in index["runs"])


def test_execute_stops_on_first_failure_and_indexes_attempted_runs(
    tmp_path,
    monkeypatch,
):
    module = load_benchmark_matrix_module()
    calls = []
    return_codes = iter([0, 7])

    def fake_run(command, check):
        calls.append((command, check))
        return subprocess.CompletedProcess(command, next(return_codes))

    replace_subprocess_run(module, monkeypatch, fake_run)

    result = module.main(execution_argv(tmp_path, "--execute"))

    assert result == 7
    assert len(calls) == 2
    assert [option_value(command, "--backend") for command, _ in calls] == [
        "eager",
        "eager",
    ]
    index = json.loads(
        (tmp_path / "matrix" / "matrix.index.json").read_text()
    )
    assert [run["return_code"] for run in index["runs"]] == [0, 7]
    assert [run["command"] for run in index["runs"]] == [
        command for command, _ in calls
    ]
