#!/usr/bin/env bash
set -euo pipefail

venv_path="${FIREREDLID_VENV_PATH:-/opt/.venv}"
model_dir="${FIREREDLID_MODEL_DIR:-/opt/FireRedLID}"
export TENSORRT_ROOT="${TENSORRT_ROOT:-/opt/tensorrt}"
export PYTORCH_CUDA_LIB="${PYTORCH_CUDA_LIB:-${venv_path}/lib/python3.12/site-packages/nvidia/cu13/lib}"

if [[ ! -f "${venv_path}/bin/activate" ]]; then
    echo "missing Python environment: ${venv_path}" >&2
    exit 1
fi
if [[ ! -d "${PYTORCH_CUDA_LIB}" ]]; then
    echo "missing PyTorch CUDA libraries: ${PYTORCH_CUDA_LIB}" >&2
    exit 1
fi

# Keep PyTorch's matching libcublas/libcublasLt pair ahead of system CUDA.
export LD_LIBRARY_PATH="${PYTORCH_CUDA_LIB}:${TENSORRT_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
source "${venv_path}/bin/activate"

if [[ $# -eq 0 ]]; then
    set -- --backend eager --use-gpu --use-half --port 12345
fi

if [[ "${1}" == -* ]]; then
    has_model_dir=0
    for arg in "$@"; do
        case "${arg}" in
            --model-dir | --model-dir=*)
                has_model_dir=1
                break
                ;;
        esac
    done
    if [[ ${has_model_dir} -eq 0 ]]; then
        set -- --model-dir "${model_dir}" "$@"
    fi
    exec fireredlid-server "$@"
fi

exec "$@"
