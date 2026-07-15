import csv
import hashlib
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

import torch


NVIDIA_SMI_COMMAND = [
    "nvidia-smi",
    "--query-gpu=index,name,uuid,driver_version,pci.bus_id,memory.total",
    "--format=csv,noheader,nounits",
]
NVIDIA_SMI_FIELDS = [
    "index",
    "name",
    "uuid",
    "driver_version",
    "pci_bus_id",
    "memory_total_mib",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_artifact(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"input artifact not found: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def model_input_artifacts(model_dir):
    model_dir = Path(model_dir)
    return {
        "checkpoint": file_artifact(model_dir / "model.pth.tar"),
        "cmvn": file_artifact(model_dir / "cmvn.ark"),
        "dictionary": file_artifact(model_dir / "dict.txt"),
    }


def engine_input_artifacts(engine_dir):
    engine_dir = Path(engine_dir)
    return {
        "engine": file_artifact(engine_dir / "encoder.plan"),
        "engine_manifest": file_artifact(engine_dir / "manifest.json"),
        "profiles": file_artifact(engine_dir / "profiles.yaml"),
    }


def _external_onnx_locations(onnx_path):
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    return sorted(
        {
            item.value
            for initializer in model.graph.initializer
            for item in initializer.external_data
            if item.key == "location"
        }
    )


def onnx_bundle_artifact(onnx_path):
    onnx_path = Path(onnx_path).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"input artifact not found: {onnx_path}")

    locations = _external_onnx_locations(onnx_path)
    files = [(onnx_path.name, onnx_path)]
    for location in locations:
        relative_path = Path(location)
        if relative_path.is_absolute():
            raise ValueError(
                "absolute external ONNX data location is not allowed: "
                f"{location}"
            )
        path = (onnx_path.parent / relative_path).resolve()
        try:
            path.relative_to(onnx_path.parent)
        except ValueError as error:
            raise ValueError(
                "external ONNX data location escapes graph directory: "
                f"{location}"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(f"missing external ONNX data: {location}")
        files.append((location, path))

    bundle_digest = hashlib.sha256()
    file_reports = []
    for label, path in files:
        bundle_digest.update(label.encode("utf-8"))
        bundle_digest.update(b"\0")
        file_digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                bundle_digest.update(chunk)
                file_digest.update(chunk)
        file_reports.append(
            {"path": label, "sha256": file_digest.hexdigest()}
        )

    return {
        "path": str(onnx_path),
        "sha256": bundle_digest.hexdigest(),
        "file_count": len(files),
        "external_file_count": len(locations),
        "files": file_reports,
    }


def _git_value(repo_root, *arguments):
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=Path(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def collect_git_info(repo_root):
    return {
        "branch": _git_value(repo_root, "branch", "--show-current"),
        "commit": _git_value(repo_root, "rev-parse", "HEAD"),
    }


def _package_version(*distribution_names):
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _compute_capability(device):
    try:
        major, minor = torch.cuda.get_device_capability(device)
        return f"{major}.{minor}"
    except Exception:
        try:
            properties = torch.cuda.get_device_properties(device)
            return f"{properties.major}.{properties.minor}"
        except Exception:
            return None


def collect_nvidia_smi():
    result = {
        "available": False,
        "command": list(NVIDIA_SMI_COMMAND),
        "return_code": None,
        "devices": [],
    }
    try:
        completed = subprocess.run(
            NVIDIA_SMI_COMMAND,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return result
    result["return_code"] = completed.returncode
    if completed.returncode != 0:
        return result
    result["available"] = True
    result["devices"] = [
        dict(zip(NVIDIA_SMI_FIELDS, row))
        for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True)
        if row
    ]
    return result


def collect_cuda_info():
    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        available = False
    try:
        device_count = int(torch.cuda.device_count()) if available else 0
    except Exception:
        device_count = 0

    devices = []
    for device in range(device_count):
        try:
            name = torch.cuda.get_device_name(device)
        except Exception:
            name = None
        devices.append(
            {
                "index": device,
                "name": name,
                "compute_capability": _compute_capability(device),
            }
        )

    try:
        current_device = int(torch.cuda.current_device()) if available else None
    except Exception:
        current_device = None
    current = next(
        (item for item in devices if item["index"] == current_device),
        None,
    )
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "available": available,
        "device_count": device_count,
        "current_device": current_device,
        "gpu_name": current["name"] if current else None,
        "compute_capability": (
            current["compute_capability"] if current else None
        ),
        "devices": devices,
        "nvidia_smi": collect_nvidia_smi(),
    }


def _json_primitive(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_primitive(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_primitive(item) for item in value]
    if isinstance(value, set):
        return [_json_primitive(item) for item in sorted(value, key=str)]
    return str(value)


def collect_provenance(arguments, input_artifacts, repo_root):
    return {
        "schema_version": 1,
        "git": collect_git_info(repo_root),
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "versions": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "tensorrt": _package_version("tensorrt"),
            "onnx": _package_version("onnx"),
            "onnxruntime": _package_version("onnxruntime"),
        },
        "cuda": collect_cuda_info(),
        "input_artifacts": _json_primitive(input_artifacts),
        "arguments": _json_primitive(arguments),
    }
